#!/usr/bin/env python3
"""Check problems.jsonl: validate each problem, recompute its answer with
circuits.py and compare against the stored one. See --help for the options.

Exit status is 0 if everything passed, 1 if a problem failed or the file is
broken, 2 for a bad command line (unknown id or topic, unreadable file, a
value that isn't a number).
"""

import argparse
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import circuits  # noqa: E402

DEFAULT_FILE = os.path.join(HERE, "problems.jsonl")

REQUIRED = ("id", "topic", "difficulty", "statement", "method", "params", "answer",
            "unit", "rel_tol", "solution", "common_mistakes", "criteria")
OPTIONAL = ("abs_tol",)
MISTAKE_KEYS = {"mistake", "why", "wrong_answer"}
CRITERION_KEYS = {"id", "text", "weight"}

UNIT_PREFIX = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "": 1.0, "k": 1e3, "M": 1e6, "G": 1e9}
UNIT_BASES = ("V", "A", "W", "VA", "var", "F", "H", "Hz", "s", "ohm", "deg", "pu", "J", "ratio")
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class UsageError(Exception):
    """Bad command line input (exit status 2)."""


def unit_scale(unit):
    """'mA' -> 1e-3, 'kvar' -> 1e3, 'V' -> 1.0; None if not recognised."""
    if not isinstance(unit, str):
        return None
    if unit in UNIT_BASES:
        return 1.0
    if unit[:1] in UNIT_PREFIX and unit[1:] in UNIT_BASES:
        return UNIT_PREFIX[unit[:1]]
    return None


def is_number(x):
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        return False
    try:
        return math.isfinite(x)
    except OverflowError:          # JSON happily gives us ints like 10**400
        return False


def within(candidate, answer, rel_tol, abs_tol=0.0):
    return abs(candidate - answer) <= rel_tol * abs(answer) + abs_tol


def _abs_tol(p):
    t = p.get("abs_tol", 0.0)
    return t if is_number(t) and t >= 0 else 0.0


def split_steps(solution):
    """Split '1. foo 2. bar 3. baz' into ['1. foo', '2. bar', '3. baz']."""
    if not isinstance(solution, str) or not solution.lstrip().startswith("1. "):
        return [solution] if isinstance(solution, str) else []
    text = solution.strip()
    starts = [0]
    k = 2
    while True:
        i = text.find(" %d. " % k, starts[-1])
        if i < 0:
            break
        starts.append(i + 1)
        k += 1
    starts.append(len(text) + 1)
    return [text[starts[j]:starts[j + 1] - 1].strip() for j in range(len(starts) - 1)]


def _reject_constant(name):
    raise ValueError("%s is not allowed in JSON numbers" % name)


def load_problems(path):
    """Returns (entries, errors); entries is a list of (line_number, dict)."""
    entries, errors = [], []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError as exc:
        raise UsageError("cannot read %s: %s" % (path, exc.strerror or exc))
    except UnicodeDecodeError:
        raise UsageError("%s is not valid UTF-8" % path)
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line, parse_constant=_reject_constant)
        except Exception as exc:                       # includes RecursionError
            errors.append("line %d: invalid JSON (%s)" % (n, str(exc).splitlines()[0][:80]))
            continue
        if not isinstance(obj, dict):
            errors.append("line %d: expected a JSON object" % n)
            continue
        entries.append((n, obj))
    if not entries and not errors:
        errors.append("no problems found in %s" % path)
    return entries, errors


def validate_schema(p):
    errs = []
    for key in REQUIRED:
        if key not in p:
            errs.append("missing field %r" % key)
    for key in p:
        if key not in REQUIRED and key not in OPTIONAL:
            errs.append("unknown field %r" % key)
    if errs:
        return errs        # the checks below assume every field is there

    if not isinstance(p["id"], str) or not ID_PATTERN.match(p["id"]):
        errs.append("id must be a string of letters, digits, '_', '-', '.'")
    if not isinstance(p["topic"], str) or not p["topic"].strip():
        errs.append("topic must be a non-empty string")
    d = p["difficulty"]
    if isinstance(d, bool) or not isinstance(d, int) or not 1 <= d <= 3:
        errs.append("difficulty must be an integer 1, 2 or 3")
    if not isinstance(p["statement"], str) or len(p["statement"].strip()) < 20:
        errs.append("statement must be a string of at least 20 characters")
    if not isinstance(p["method"], str) or not p["method"]:
        errs.append("method must be a non-empty string")
    else:
        try:
            circuits.get_method(p["method"])
        except KeyError:
            errs.append("unknown method %r" % p["method"])
    if not isinstance(p["params"], dict) or not all(isinstance(k, str) for k in p["params"]):
        errs.append("params must be an object")
    if not is_number(p["answer"]):
        errs.append("answer must be a finite number")
    if not isinstance(p["unit"], str) or not p["unit"].strip():
        errs.append("unit must be a non-empty string")
    elif unit_scale(p["unit"]) is None:
        errs.append("unrecognised unit %r" % p["unit"])
    rt = p["rel_tol"]
    if not is_number(rt) or not 0 < rt <= 0.5:
        errs.append("rel_tol must be a number in (0, 0.5]")
    if "abs_tol" in p and (not is_number(p["abs_tol"]) or p["abs_tol"] < 0):
        errs.append("abs_tol must be a number >= 0")
    sol = p["solution"]
    if not isinstance(sol, str) or not sol.lstrip().startswith("1. "):
        errs.append("solution must be a string of numbered steps starting '1. '")
    elif len(split_steps(sol)) < 2:
        errs.append("solution needs at least two numbered steps")

    cm = p["common_mistakes"]
    if not isinstance(cm, list) or not 1 <= len(cm) <= 3:
        errs.append("common_mistakes must be a list of 1 to 3 items")
    else:
        for i, m in enumerate(cm):
            if not isinstance(m, dict) or not MISTAKE_KEYS >= set(m) or \
                    not isinstance(m.get("mistake"), str) or not m.get("mistake", "").strip() or \
                    not isinstance(m.get("why"), str) or not m.get("why", "").strip():
                errs.append("common_mistakes[%d] needs non-empty 'mistake' and 'why' (optional 'wrong_answer')" % i)
                continue
            w = m.get("wrong_answer")
            if w is not None:
                if not is_number(w):
                    errs.append("common_mistakes[%d].wrong_answer must be a number or null" % i)
                elif is_number(p["answer"]) and is_number(rt) and within(w, p["answer"], rt, _abs_tol(p)):
                    errs.append("common_mistakes[%d].wrong_answer %r is inside the tolerance of the correct answer" % (i, w))

    cr = p["criteria"]
    if not isinstance(cr, list) or not 3 <= len(cr) <= 5:
        errs.append("criteria must be a list of 3 to 5 items")
    else:
        seen = set()
        for i, c in enumerate(cr):
            if not isinstance(c, dict) or set(c) != CRITERION_KEYS:
                errs.append("criteria[%d] must have exactly id, text, weight" % i)
                continue
            if not isinstance(c["id"], str) or not c["id"].strip():
                errs.append("criteria[%d].id must be a non-empty string" % i)
            elif c["id"] in seen:
                errs.append("criteria[%d].id %r is repeated" % (i, c["id"]))
            else:
                seen.add(c["id"])
            if not isinstance(c["text"], str) or not c["text"].strip():
                errs.append("criteria[%d].text must be a non-empty string" % i)
            if not is_number(c["weight"]) or not c["weight"] > 0:
                errs.append("criteria[%d].weight must be a positive number" % i)
    return errs


def compute(p):
    """Recompute the answer in the problem's unit. Returns (value, error)."""
    try:
        fn = circuits.get_method(p["method"])
        result = fn(**p["params"])
    except KeyError:
        return None, "unknown method %r" % (p.get("method"),)
    except Exception as exc:                   # any bad params must not crash the run
        msg = str(exc).splitlines()[0][:120] if str(exc) else ""
        return None, "method %s raised %s: %s" % (p.get("method"), type(exc).__name__, msg)
    if isinstance(result, bool) or not hasattr(result, "__float__") or isinstance(result, complex):
        return None, "method returned %r, not a real number" % (result,)
    try:
        value = float(result) / unit_scale(p["unit"])
    except Exception as exc:
        return None, "cannot convert result: %s" % exc
    if not math.isfinite(value):
        return None, "method returned a non-finite value"
    return value, None


def check_problem(p):
    """Returns (ok, message, computed_value)."""
    errs = validate_schema(p)
    if errs:
        return False, "; ".join(errs), None
    value, err = compute(p)
    if err:
        return False, err, None
    if not within(value, p["answer"], p["rel_tol"], _abs_tol(p)):
        return False, "stored answer %s %s but recomputed %.6g (rel_tol %g)" % (
            p["answer"], p["unit"], value, p["rel_tol"]), value
    return True, "", value


def _find(entries, pid):
    for _, p in entries:
        if p.get("id") == pid:
            return p
    raise UsageError("no problem with id %r (try --list)" % pid)


def cmd_verify(entries, load_errors, topic, quiet, out):
    failures = list(load_errors)
    for msg in load_errors:
        print("FAIL  %s" % msg, file=out)
    ids = {}
    for n, p in entries:
        pid = p.get("id")
        if isinstance(pid, str):
            if pid in ids:
                msg = "line %d: duplicate id %r (first on line %d)" % (n, pid, ids[pid])
                failures.append(msg)
                print("FAIL  %s" % msg, file=out)
            else:
                ids[pid] = n
    selected = [(n, p) for n, p in entries
                if topic is None or p.get("topic") == topic]
    if topic is not None and not selected:
        topics = sorted(set(str(p.get("topic")) for _, p in entries))
        raise UsageError("no problems with topic %r; topics: %s" % (topic, ", ".join(topics)))
    passed = 0
    for n, p in selected:
        ok, msg, value = check_problem(p)
        label = p.get("id", "?") if isinstance(p.get("id"), str) else "line %d" % n
        if ok:
            passed += 1
            if not quiet:
                print("ok    %-34s %-13s %s %s" % (label, p["topic"], p["answer"], p["unit"]), file=out)
        else:
            failures.append("%s: %s" % (label, msg))
            print("FAIL  %s: %s" % (label, msg), file=out)
    topics = sorted(set(str(p.get("topic")) for _, p in selected))
    print("%d problem(s) checked, %d passed, %d failure(s); topics: %s" % (
        len(selected), passed, len(failures), ", ".join(topics) or "-"), file=out)
    return 1 if failures else 0


def cmd_show(entries, pid, out):
    p = _find(entries, pid)
    print("%s   [%s, difficulty %s]" % (p.get("id"), p.get("topic"), p.get("difficulty")), file=out)
    print(file=out)
    print(p.get("statement", ""), file=out)
    print(file=out)
    print("Solution:", file=out)
    for step in split_steps(p.get("solution", "")):
        print("  " + step, file=out)
    print(file=out)
    print("Answer: %s %s (relative tolerance %s)" % (p.get("answer"), p.get("unit"), p.get("rel_tol")), file=out)
    return 0


def cmd_check(entries, pid, value_text, out):
    p = _find(entries, pid)
    try:
        cand = float(value_text)
    except ValueError:
        raise UsageError("cannot read %r as a number (give it in %s)" % (value_text, p.get("unit")))
    if not math.isfinite(cand):
        raise UsageError("candidate value must be finite")
    errs = validate_schema(p)
    if errs:
        raise UsageError("problem %r is invalid: %s" % (pid, "; ".join(errs)))
    ans, unit, rt = p["answer"], p["unit"], p["rel_tol"]
    abs_tol = _abs_tol(p)
    if within(cand, ans, rt, abs_tol):
        print("PASS: %g %s is within %g%% of %g %s" % (cand, unit, rt * 100, ans, unit), file=out)
        return 0
    print("FAIL: %g %s is outside %g%% of the reference answer %g %s" % (cand, unit, rt * 100, ans, unit), file=out)
    for m in p["common_mistakes"]:
        w = m.get("wrong_answer")
        if w is not None and within(cand, w, rt, abs_tol):
            print("      this matches a listed mistake: %s" % m["mistake"], file=out)
            print("      why it is wrong: %s" % m["why"], file=out)
    return 1


def cmd_list(entries, topic, out):
    shown = 0
    for _, p in entries:
        if topic is None or p.get("topic") == topic:
            print("%-34s %-13s d%s  %s" % (p.get("id"), p.get("topic"), p.get("difficulty"), p.get("unit")), file=out)
            shown += 1
    if topic is not None and not shown:
        raise UsageError("no problems with topic %r" % topic)
    return 0


def build_parser():
    ap = argparse.ArgumentParser(
        prog="verify.py",
        description="Verify the problem bank: schema, recomputed answers, tolerances.")
    ap.add_argument("--file", default=DEFAULT_FILE, help="problem file (default: problems.jsonl next to this script)")
    ap.add_argument("--topic", help="only work on this topic")
    ap.add_argument("--show", metavar="ID", help="print a problem's statement and worked solution")
    ap.add_argument("--check", nargs=2, metavar=("ID", "VALUE"),
                    help="compare a candidate answer (in the problem's unit) with the stored answer")
    ap.add_argument("--list", action="store_true", help="list problem ids, topics and difficulties")
    ap.add_argument("-q", "--quiet", action="store_true", help="only print failures and the summary")
    return ap


def main(argv=None, out=None):
    out = sys.stdout if out is None else out
    if out is sys.stdout:
        for stream in (sys.stdout, sys.stderr):      # never die on an unencodable character
            try:
                stream.reconfigure(errors="replace")
            except Exception:
                pass
    ap = build_parser()
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:                  # argparse already printed its message
        return 2 if exc.code not in (0, None) else 0
    try:
        entries, load_errors = load_problems(args.file)
        if args.show:
            return cmd_show(entries, args.show, out)
        if args.check:
            return cmd_check(entries, args.check[0], args.check[1], out)
        if args.list:
            return cmd_list(entries, args.topic, out)
        return cmd_verify(entries, load_errors, args.topic, args.quiet, out)
    except UsageError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
