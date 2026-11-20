"""Tests for verify.py and the shipped problems.jsonl."""

import contextlib
import copy
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

import circuits
import verify

HERE = os.path.dirname(os.path.abspath(__file__))
PROBLEMS = os.path.join(HERE, "problems.jsonl")
VERIFY = os.path.join(HERE, "verify.py")


def run(args):
    """Run verify.main in-process; return (exit_code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(err):
        code = verify.main(args, out=out)
    return code, out.getvalue(), err.getvalue()


def load_shipped():
    with open(PROBLEMS, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class TempFileCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def write(self, problems, extra_lines=()):
        path = os.path.join(self._tmp.name, "p.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for p in problems:
                fh.write((p if isinstance(p, str) else json.dumps(p, ensure_ascii=False)) + "\n")
            for line in extra_lines:
                fh.write(line + "\n")
        return path


class ShippedProblemTests(unittest.TestCase):
    def test_shipped_problems_pass(self):
        code, out, err = run([])
        self.assertEqual(code, 0, out)
        self.assertIn("0 failure(s)", out)
        self.assertEqual(err, "")

    def test_size_and_coverage(self):
        problems = load_shipped()
        self.assertGreaterEqual(len(problems), 32)
        topics = {p["topic"] for p in problems}
        for needed in ("resistive", "thevenin", "transients", "ac", "resonance",
                       "power_factor", "three_phase", "per_unit", "opamp"):
            self.assertIn(needed, topics)
            self.assertGreaterEqual(sum(1 for p in problems if p["topic"] == needed), 3, needed)
        self.assertEqual({p["difficulty"] for p in problems}, {1, 2, 3})
        self.assertEqual(len({p["id"] for p in problems}), len(problems))

    def test_answers_are_rounded(self):
        for p in load_shipped():
            value, err = verify.compute(p)
            self.assertIsNone(err, p["id"])
            self.assertLessEqual(abs(value - p["answer"]), 1e-3 * abs(value), p["id"])   # 4 significant figures
            self.assertLessEqual(p["rel_tol"], 0.02, p["id"])

    def test_wrong_answers_are_wrong(self):
        n = 0
        for p in load_shipped():
            for m in p["common_mistakes"]:
                if m["wrong_answer"] is not None:
                    n += 1
                    self.assertGreater(abs(m["wrong_answer"] - p["answer"]), p["rel_tol"] * abs(p["answer"]), p["id"])
        self.assertGreater(n, 40)

    def test_statements(self):
        for p in load_shipped():
            self.assertGreater(len(p["statement"]), 60, p["id"])
            self.assertTrue(p["statement"].rstrip().endswith((".", "?")), p["id"])
            total = sum(c["weight"] for c in p["criteria"])
            self.assertGreater(total, 0)

    def test_pitfalls_covered(self):
        text = " ".join(m["mistake"].lower() for p in load_shipped() for m in p["common_mistakes"])
        # rms vs peak, line vs phase, delta vs wye, current direction, milli/kilo scaling
        for phrase in ("peak", "line voltage", "sqrt(3)", "delta", "sign", "mh", "kω"):
            self.assertIn(phrase, text, phrase)


class CorruptionTests(TempFileCase):
    def test_corrupted_answer_fails(self):
        problems = load_shipped()
        problems[3]["answer"] = problems[3]["answer"] * 1.5
        path = self.write(problems)
        code, out, err = run(["--file", path])
        self.assertEqual(code, 1)
        self.assertIn(problems[3]["id"], out)
        self.assertIn("recomputed", out)
        self.assertNotIn("Traceback", out + err)

    def test_answer_inside_tolerance(self):
        problems = load_shipped()[:5]
        for p in problems:
            p["answer"] = p["answer"] * (1 + 0.4 * p["rel_tol"])
            for m in p["common_mistakes"]:
                m["wrong_answer"] = None
        path = self.write(problems)
        self.assertEqual(run(["--file", path])[0], 0)

    def test_changed_params_fail(self):
        problems = load_shipped()
        p = next(q for q in problems if q["id"] == "pu_change_base")
        p["params"]["v_new"] = 11.0e3
        self.assertEqual(run(["--file", self.write(problems)])[0], 1)


class SchemaTests(TempFileCase):
    def mutated(self, mutate, index=0):
        problems = copy.deepcopy(load_shipped()[:6])
        mutate(problems[index])
        return problems

    def assertFailsWith(self, problems, text, extra_lines=()):
        path = self.write(problems, extra_lines)
        code, out, err = run(["--file", path])
        self.assertEqual(code, 1, out)
        self.assertIn(text, out)
        self.assertNotIn("Traceback", out + err)

    def test_duplicate_id(self):
        problems = load_shipped()[:3]
        problems[1]["id"] = problems[0]["id"]
        self.assertFailsWith(problems, "duplicate id")

    def test_missing_fields(self):
        for key in verify.REQUIRED:
            problems = self.mutated(lambda p, k=key: p.pop(k))
            self.assertFailsWith(problems, "missing field '%s'" % key)

    def test_unknown_field(self):
        self.assertFailsWith(self.mutated(lambda p: p.update(colour="red")), "unknown field 'colour'")

    def test_difficulty(self):
        for bad in (0, 4, 2.5, "2", True, None):
            self.assertFailsWith(self.mutated(lambda p, b=bad: p.update(difficulty=b)), "difficulty")

    def test_unit(self):
        self.assertFailsWith(self.mutated(lambda p: p.update(unit="")), "unit")
        self.assertFailsWith(self.mutated(lambda p: p.update(unit="furlong")), "unrecognised unit")
        self.assertFailsWith(self.mutated(lambda p: p.update(unit=None)), "unit")

    def test_criteria_weights(self):
        def neg(p):
            p["criteria"][0]["weight"] = -1
        def zero(p):
            p["criteria"][1]["weight"] = 0
        def text_weight(p):
            p["criteria"][0]["weight"] = "2"
        for mutate in (neg, zero, text_weight):
            self.assertFailsWith(self.mutated(mutate), "weight must be a positive number")

    def test_criteria_structure(self):
        def dup(p):
            p["criteria"][1]["id"] = p["criteria"][0]["id"]
        self.assertFailsWith(self.mutated(dup), "is repeated")
        self.assertFailsWith(self.mutated(lambda p: p.update(criteria=p["criteria"][:2])), "3 to 5")
        self.assertFailsWith(self.mutated(lambda p: p.update(criteria=p["criteria"] * 2)), "3 to 5")
        self.assertFailsWith(self.mutated(lambda p: p["criteria"][0].pop("text")), "exactly id, text, weight")

    def test_common_mistakes(self):
        self.assertFailsWith(self.mutated(lambda p: p.update(common_mistakes=[])), "1 to 3")
        self.assertFailsWith(self.mutated(lambda p: p.update(common_mistakes=["just text"])), "common_mistakes[0]")
        def inside(p):
            p["common_mistakes"][0]["wrong_answer"] = p["answer"]
        self.assertFailsWith(self.mutated(inside), "inside the tolerance")

    def test_solution_format(self):
        self.assertFailsWith(self.mutated(lambda p: p.update(solution="no steps here at all")), "numbered steps")
        self.assertFailsWith(self.mutated(lambda p: p.update(solution="1. only one step")), "two numbered steps")

    def test_answer_and_tolerance_types(self):
        for bad in ("5", None, True, [1]):
            self.assertFailsWith(self.mutated(lambda p, b=bad: p.update(answer=b)), "answer must be a finite number")
        # json.dumps writes Infinity/NaN for these; the loader must refuse them
        for const in (float("inf"), float("nan")):
            line = json.dumps(dict(load_shipped()[0], answer=const))
            self.assertFailsWith(load_shipped()[:2], "invalid JSON", extra_lines=[line])
        for bad in (0, -0.1, 0.9, "1%"):
            self.assertFailsWith(self.mutated(lambda p, b=bad: p.update(rel_tol=b)), "rel_tol")

    def test_unknown_method_and_bad_params(self):
        self.assertFailsWith(self.mutated(lambda p: p.update(method="no_such_method")), "unknown method")
        self.assertFailsWith(self.mutated(lambda p: p.update(method="os")), "unknown method")
        self.assertFailsWith(self.mutated(lambda p: p["params"].update(bogus=1)), "TypeError")
        problems = self.mutated(lambda p: p.update(params={}))
        self.assertFailsWith(problems, "TypeError")
        self.assertFailsWith(self.mutated(lambda p: p.update(params=[1, 2])), "params")

    def test_method_raises(self):
        problems = load_shipped()
        p = next(q for q in problems if q["id"] == "res_divider_loaded")
        p["params"]["netlist"] = [["R", "R1", "a", "b", 10]]          # no source, no ground
        self.assertFailsWith(problems, "CircuitError")
        p["params"]["netlist"] = "not a netlist"
        self.assertFailsWith(problems, "CircuitError")
        p["params"]["measure_expr"] = 7
        self.assertFailsWith(problems, "res_divider_loaded")

    def test_floating_circuit(self):
        problems = load_shipped()
        p = next(q for q in problems if q["id"] == "res_divider_loaded")
        p["params"]["netlist"].append(["R", "Rx", "x", "y", 100])
        self.assertFailsWith(problems, "floating")

    def test_bad_json_lines(self):
        self.assertFailsWith(load_shipped()[:2], "invalid JSON", extra_lines=["{not json"])
        self.assertFailsWith(load_shipped()[:2], "expected a JSON object", extra_lines=["[1, 2, 3]"])
        self.assertFailsWith(load_shipped()[:2], "invalid JSON", extra_lines=['{"id": NaN}'])
        self.assertFailsWith(load_shipped()[:2], "invalid JSON", extra_lines=["[" * 100000])

    def test_empty_file(self):
        path = self.write([])
        code, out, err = run(["--file", path])
        self.assertEqual(code, 1)
        self.assertIn("no problems", out)

    def test_non_utf8_file(self):
        path = os.path.join(self._tmp.name, "bad.jsonl")
        with open(path, "wb") as fh:
            fh.write(b"\xff\xfe\x00bad")
        code, out, err = run(["--file", path])
        self.assertEqual(code, 2)
        self.assertIn("UTF-8", err)

    def test_missing_file(self):
        code, out, err = run(["--file", os.path.join(self._tmp.name, "nope.jsonl")])
        self.assertEqual(code, 2)
        self.assertIn("cannot read", err)
        self.assertNotIn("Traceback", err)


class CommandLineTests(TempFileCase):
    def test_show(self):
        code, out, err = run(["--show", "res_divider_loaded"])
        self.assertEqual(code, 0)
        self.assertLess(out.index("Find the voltage at OUT"), out.index("Solution:"))
        self.assertLess(out.index("Solution:"), out.index("1. The load is in parallel"))
        self.assertIn("\n  2. ", out)
        self.assertIn("Answer: 14.25 V", out)

    def test_show_unknown_id(self):
        code, out, err = run(["--show", "nope"])
        self.assertEqual(code, 2)
        self.assertIn("no problem with id", err)
        self.assertNotIn("Traceback", err)

    def test_check_pass_and_fail(self):
        code, out, _ = run(["--check", "res_divider_loaded", "14.3"])
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("PASS"))
        code, out, _ = run(["--check", "res_divider_loaded", "14.6"])
        self.assertEqual(code, 1)
        self.assertTrue(out.startswith("FAIL"))
        code, out, _ = run(["--check", "res_divider_loaded", "1e1"])
        self.assertEqual(code, 1)

    def test_check_reports_matching_mistake(self):
        code, out, _ = run(["--check", "res_divider_loaded", "16.33"])
        self.assertEqual(code, 1)
        self.assertIn("matches a listed mistake", out)
        self.assertIn("Ignored the load", out)

    def test_check_negative_answers(self):
        self.assertEqual(run(["--check", "res_mesh_direction", "-0.5"])[0], 0)
        self.assertEqual(run(["--check", "res_mesh_direction", "0.5"])[0], 1)      # sign matters
        self.assertEqual(run(["--check", "opamp_summing", "-5.02"])[0], 0)

    def test_check_bad_values(self):
        for bad in ("abc", "nan", "inf", "", "1,5"):
            code, out, err = run(["--check", "res_divider_loaded", bad])
            self.assertEqual(code, 2, repr(bad))
            self.assertNotIn("Traceback", err)
        self.assertEqual(run(["--check", "nope", "1"])[0], 2)

    def test_topic_filter(self):
        code, out, err = run(["--topic", "opamp"])
        self.assertEqual(code, 0)
        self.assertIn("opamp_summing", out)
        self.assertNotIn("res_divider_loaded", out)
        self.assertIn("topics: opamp", out)
        code, out, err = run(["--topic", "nonsense"])
        self.assertEqual(code, 2)
        self.assertIn("topics:", err)

    def test_topic_filter_failures(self):
        problems = load_shipped()
        for p in problems:
            if p["id"] == "opamp_summing":
                p["answer"] = 99.0
        path = self.write(problems)
        self.assertEqual(run(["--topic", "opamp", "--file", path])[0], 1)
        self.assertEqual(run(["--topic", "resistive", "--file", path])[0], 0)

    def test_list_and_quiet(self):
        code, out, _ = run(["--list"])
        self.assertEqual(code, 0)
        self.assertEqual(len([l for l in out.splitlines() if l.strip()]), len(load_shipped()))
        code, out, _ = run(["-q"])
        self.assertEqual(code, 0)
        self.assertEqual(len(out.strip().splitlines()), 1)

    def test_usage_error_exit_code(self):
        code, out, err = run(["--no-such-flag"])
        self.assertEqual(code, 2)
        self.assertNotIn("Traceback", err)

    def test_subprocess_runs_clean(self):
        r = subprocess.run([sys.executable, VERIFY], capture_output=True, text=True, cwd=self._tmp.name)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(r.stderr, "")
        r = subprocess.run([sys.executable, VERIFY, "--file", "missing.jsonl"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertNotIn("Traceback", r.stderr)
        r = subprocess.run([sys.executable, VERIFY, "--check", "res_divider_loaded", "x"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertNotIn("Traceback", r.stderr)


class HelperTests(unittest.TestCase):
    def test_unit_scale(self):
        self.assertEqual(verify.unit_scale("V"), 1.0)
        self.assertEqual(verify.unit_scale("mA"), 1e-3)
        self.assertEqual(verify.unit_scale("uF"), 1e-6)
        self.assertEqual(verify.unit_scale("kvar"), 1e3)
        self.assertEqual(verify.unit_scale("kHz"), 1e3)
        self.assertEqual(verify.unit_scale("ohm"), 1.0)
        for bad in ("", "x", "kk", "mm", None, 5, "kratio2"):
            self.assertIsNone(verify.unit_scale(bad), repr(bad))

    def test_within(self):
        self.assertTrue(verify.within(101, 100, 0.01))
        self.assertFalse(verify.within(101.1, 100, 0.01))
        self.assertTrue(verify.within(-0.499, -0.5, 0.01))
        self.assertTrue(verify.within(1e-12, 0, 0.01, abs_tol=1e-9))
        self.assertFalse(verify.within(1e-3, 0, 0.5))

    def test_split_steps(self):
        self.assertEqual(verify.split_steps("1. a b 2. c 3. d = 2.5 V"), ["1. a b", "2. c", "3. d = 2.5 V"])
        self.assertEqual(verify.split_steps("1. only"), ["1. only"])

    def test_compute_converts_units(self):
        p = {"method": "series_rlc_impedance_abs", "params": {"r": 10, "l": 1e-3, "c": 1e-6, "f": 1000.0},
             "unit": "kohm"}
        value, err = verify.compute(p)
        self.assertIsNone(err)
        expected = abs(circuits.series_rlc_impedance(10, 1e-3, 1e-6, 1000.0)) / 1e3
        self.assertAlmostEqual(value, expected, places=12)

    def test_compute_non_real(self):
        circuits.METHODS["_returns_complex"] = lambda: 1 + 2j
        circuits.METHODS["_returns_text"] = lambda: "5"
        circuits.METHODS["_returns_nan"] = lambda: float("nan")
        circuits.METHODS["_returns_bool"] = lambda: True
        self.addCleanup(lambda: [circuits.METHODS.pop(k) for k in
                                 ("_returns_complex", "_returns_text", "_returns_nan", "_returns_bool")])
        for name in ("_returns_complex", "_returns_text", "_returns_nan", "_returns_bool"):
            value, err = verify.compute({"method": name, "params": {}, "unit": "V"})
            self.assertIsNone(value, name)
            self.assertIsNotNone(err, name)


class JunkInputTests(TempFileCase):
    JUNK = [None, True, 0, -1, 10 ** 400, -10 ** 400, 1e308, "x", "", [], {}, [None], [[]], {"a": None}, "\u00e9" * 40, 0.5]

    def test_junk_fields(self):
        base = load_shipped()
        for p in [base[0], next(q for q in base if q["id"] == "pu_change_base")]:
            for key in p:
                for junk in self.JUNK:
                    q = copy.deepcopy(p)
                    q[key] = junk
                    path = self.write([q])
                    for args in (["--file", path], ["--file", path, "--list"],
                                 ["--file", path, "--show", p["id"]], ["--file", path, "--check", p["id"], "1"]):
                        code, out, err = run(args)               # must return, never raise
                        self.assertIn(code, (0, 1, 2))
                        self.assertNotIn("Traceback", out + err)
                    self.assertIn(run(["--file", path, "-q"])[0], (0, 1))

    def test_junk_params(self):
        for p in load_shipped():
            for key in p["params"]:
                for junk in self.JUNK:
                    q = copy.deepcopy(p)
                    q["params"][key] = junk
                    value, err = verify.compute(q)
                    self.assertTrue(value is None or isinstance(value, float))
                    self.assertTrue(value is not None or isinstance(err, str))

    def test_huge_integers_in_json(self):
        problems = load_shipped()[:2]
        line = json.dumps(problems[0]).replace('"answer": %s' % json.dumps(problems[0]["answer"]), '"answer": 1' + "0" * 400)
        path = self.write([problems[1]], [line])
        code, out, err = run(["--file", path])
        self.assertEqual(code, 1)
        self.assertNotIn("Traceback", out + err)

    def test_directory_as_file(self):
        code, out, err = run(["--file", self._tmp.name])
        self.assertEqual(code, 2)
        self.assertNotIn("Traceback", err)


if __name__ == "__main__":
    unittest.main()
