"""Tests for circuits.py.  Expected values are derived by hand or by a route
that does not use the code under test (closed forms, Cramer's rule, brute force
scans, a small time-domain simulation)."""

import cmath
import math
import unittest
from fractions import Fraction as F

import circuits as C
from circuits import Circuit, CircuitError, SingularCircuitError

TWO_PI = 2 * math.pi


def close(a, b, rel=1e-9, abs_=1e-12):
    return abs(a - b) <= rel * max(abs(a), abs(b)) + abs_


class DCSolverTests(unittest.TestCase):
    def test_divider_exact_and_float(self):
        c = Circuit()
        c.voltage_source("V1", "in", "0", 12)
        c.resistor("R1", "in", "out", 1000)
        c.resistor("R2", "out", "0", 3000)
        s = c.solve_dc()
        self.assertEqual(s.v("out"), F(9))                       # 12 * 3/4, exact
        self.assertIsInstance(s.v("out"), F)
        cf = Circuit()
        cf.voltage_source("V1", "in", "0", 12.0)
        cf.resistor("R1", "in", "out", 1000.0)
        cf.resistor("R2", "out", "0", 3000.0)
        sf = cf.solve_dc()
        self.assertIsInstance(sf.v("out"), float)
        self.assertAlmostEqual(sf.v("out"), 9.0, places=12)

    def test_voltage_source_current_sign(self):
        # source delivering power has a negative current (first node -> second node inside it)
        c = Circuit()
        c.voltage_source("V1", "a", "0", 10)
        c.resistor("R1", "a", "0", 5)
        s = c.solve_dc()
        self.assertEqual(s.i("V1"), F(-2))
        self.assertEqual(s.i("R1"), F(2))
        self.assertEqual(s.power("V1"), F(-20))                    # delivers 20 W
        self.assertEqual(s.power("R1"), F(20))

    def test_current_source_into_parallel_resistors(self):
        c = Circuit()
        c.current_source("I1", "0", "a", F(3, 1000))              # pushes 3 mA into a
        c.resistor("R1", "a", "0", 1000)
        c.resistor("R2", "a", "0", 2000)
        # hand: Rp = 2000/3 ohm, v = 3e-3 * 2000/3 = 2 V
        self.assertEqual(c.solve_dc().v("a"), F(2))

    def test_wheatstone_bridge_hand_derived(self):
        # R1=1k (P-A), R2=2k (A-0), R3=3k (P-B), R4=4k (B-0), 10 V, detector 500 ohm A-B.
        # Hand: vA = 20/3, vB = 40/7, Vth = 20/21 V, Rth = 2000/3 + 12000/7 = 50000/21 ohm
        # I = Vth/(Rth + 500) = (20/21)/(60500/21) = 1/3025 A
        c = Circuit()
        c.voltage_source("V1", "p", "0", 10)
        c.resistor("R1", "p", "a", 1000)
        c.resistor("R2", "a", "0", 2000)
        c.resistor("R3", "p", "b", 3000)
        c.resistor("R4", "b", "0", 4000)
        c.resistor("Rd", "a", "b", 500)
        s = c.solve_dc()
        self.assertEqual(s.i("Rd"), F(1, 3025))
        # unloaded bridge: balanced when R1*R4 = R2*R3 (here it is not)
        c2 = c.without("Rd")
        s2 = c2.solve_dc()
        self.assertEqual(s2.v("a"), F(20, 3))
        self.assertEqual(s2.v("b"), F(40, 7))
        self.assertEqual(s2.v("a", "b"), F(20, 21))

    def test_balanced_bridge(self):
        c = Circuit()
        c.voltage_source("V1", "p", "0", 7)
        c.resistor("R1", "p", "a", 100)
        c.resistor("R2", "a", "0", 200)
        c.resistor("R3", "p", "b", 300)
        c.resistor("R4", "b", "0", 600)        # R1/R2 = R3/R4
        c.resistor("Rd", "a", "b", 47)
        self.assertEqual(c.solve_dc().i("Rd"), 0)

    def test_superposition(self):
        # V1=10 V -R1=2- x ; V2=4 V -R2=3- x ; R3=6 from x to ground.
        # Source 1 alone: 10 * (3||6)/(2 + 3||6) = 10*2/4 = 5
        # Source 2 alone: 4 * (2||6)/(3 + 2||6) = 4*1.5/4.5 = 4/3
        # Sum = 19/3.  Node equation check: (10-x)/2 + (4-x)/3 = x/6 -> 38 = 6x.
        c = Circuit()
        c.voltage_source("V1", "n1", "0", 10)
        c.voltage_source("V2", "n2", "0", 4)
        c.resistor("R1", "n1", "x", 2)
        c.resistor("R2", "n2", "x", 3)
        c.resistor("R3", "x", "0", 6)
        expected = F(5) + F(4, 3)
        self.assertEqual(expected, F(19, 3))
        self.assertEqual(c.solve_dc().v("x"), expected)
        # each source alone, other one replaced by a short
        a = Circuit()
        a.voltage_source("V1", "n1", "0", 10)
        a.voltage_source("V2", "n2", "0", 0)
        a.resistor("R1", "n1", "x", 2); a.resistor("R2", "n2", "x", 3); a.resistor("R3", "x", "0", 6)
        self.assertEqual(a.solve_dc().v("x"), F(5))

    def test_branch_current_direction(self):
        # 24 V - 6 ohm and 5 V - 4 ohm branches into node x, 3 ohm to ground.
        # 2(24 - x) + 3(5 - x) = 4x  ->  x = 7 V;  current out of the 5 V source = (5-7)/4 = -0.5 A
        c = Circuit()
        c.voltage_source("V1", "a", "0", 24)
        c.voltage_source("V2", "b", "0", 5)
        c.resistor("R1", "a", "x", 6)
        c.resistor("R2", "b", "x", 4)
        c.resistor("R3", "x", "0", 3)
        s = c.solve_dc()
        self.assertEqual(s.v("x"), F(7))
        self.assertEqual(s.i("V2"), F(1, 2))                       # flows into the + terminal
        self.assertEqual(s.i("R2"), F(-1, 2))                      # b -> x

    def test_bridge_delta_wye(self):
        # R1=10 (A-C), R2=30 (A-D), R3=40 (C-B), R4=20 (D-B), R5=15 (C-D)
        s = F(10 + 30 + 15)
        ra, rc, rd = F(300) / s, F(150) / s, F(450) / s
        req = ra + (rc + 40) * (rd + 20) / ((rc + 40) + (rd + 20))
        c = Circuit()
        for name, a, b, r in [("R1", "A", "C", 10), ("R2", "A", "D", 30), ("R3", "C", "B", 40),
                              ("R4", "D", "B", 20), ("R5", "C", "D", 15)]:
            c.resistor(name, a, b, r)
        self.assertEqual(C.thevenin(c, "A", "B").z_th, req)
        self.assertEqual(req, F(875, 39))                              # 22.4359 ohm

    def test_dc_inductor_and_capacitor(self):
        c = Circuit()
        c.voltage_source("V1", "a", "0", 6)
        c.resistor("R1", "a", "b", 2)
        c.inductor("L1", "b", "c", F(1, 10))
        c.resistor("R2", "c", "0", 1)
        c.capacitor("C1", "c", "0", F(1, 1000))
        s = c.solve_dc()
        self.assertEqual(s.v("b"), s.v("c"))                       # inductor is a short
        self.assertEqual(s.i("L1"), F(2))                          # 6 V / 3 ohm
        self.assertEqual(s.i("C1"), 0)                             # capacitor is open

    def test_exact_vs_float(self):
        rows = [["V", "V1", "a", "0", 5], ["R", "R1", "a", "b", 330], ["R", "R2", "b", "0", 470],
                ["R", "R3", "b", "c", 1000], ["R", "R4", "c", "0", 220], ["I", "I1", "0", "c", 0.001]]
        ex = Circuit.from_netlist(rows, exact=True).solve_dc()
        fl = Circuit.from_netlist([r[:4] + [float(r[4])] for r in rows]).solve_dc()
        for n in ("a", "b", "c"):
            self.assertIsInstance(ex.v(n), F)
            self.assertTrue(close(float(ex.v(n)), fl.v(n), 1e-12))

    def test_kcl(self):
        c = Circuit.from_netlist([["V", "V1", "a", "0", 9], ["R", "R1", "a", "b", 100],
                                  ["R", "R2", "b", "0", 200], ["R", "R3", "b", "c", 300],
                                  ["R", "R4", "c", "0", 400], ["I", "I1", "0", "c", 0.01]], exact=True)
        s = c.solve_dc()
        for node in ("a", "b", "c"):
            total = F(0)
            for el in c.elements:
                cur = s.i(el.name)
                if el.nodes[0] == node:
                    total += cur
                if el.nodes[1] == node:
                    total -= cur
            self.assertEqual(total, 0, node)


class ACSolverTests(unittest.TestCase):
    def test_series_rlc_impedance(self):
        R, L, Cc = 12.0, 3e-3, 4.7e-6
        f0 = 1 / (TWO_PI * math.sqrt(L * Cc))
        for f in (50.0, 400.0, 1000.0, f0, 5000.0, 20000.0):
            w = TWO_PI * f
            z = R + 1j * (w * L - 1 / (w * Cc))                    # hand formula
            c = Circuit()
            c.voltage_source("V1", "in", "0", 1)
            c.resistor("R1", "in", "a", R)
            c.inductor("L1", "a", "b", L)
            c.capacitor("C1", "b", "0", Cc)
            s = c.solve_ac(f)
            i = s.i("R1")
            self.assertTrue(abs(i - 1 / z) <= 1e-9 * abs(1 / z), f)
            # voltage across C by the divider formula
            zc = 1 / (1j * w * Cc)
            self.assertTrue(abs(s.v("b") - zc / z) <= 1e-9 * abs(zc / z), f)
        # at resonance the impedance is purely resistive
        s = c.solve_ac(f0)
        self.assertAlmostEqual(abs(1 / s.i("R1") - R), 0.0, places=6)

    def test_rc_lowpass_corner(self):
        R, Cc = 1000.0, 1e-7
        fc = 1 / (TWO_PI * R * Cc)
        c = Circuit.from_netlist([["V", "V1", "in", "0", [1, 0]], ["R", "R1", "in", "out", R],
                                  ["C", "C1", "out", "0", Cc]])
        s = c.solve_ac(fc)
        self.assertAlmostEqual(abs(s.v("out")), 1 / math.sqrt(2), places=12)
        self.assertAlmostEqual(math.degrees(cmath.phase(s.v("out"))), -45.0, places=9)
        self.assertAlmostEqual(C.measure(s, "v(out):deg"), -45.0, places=9)

    def test_parallel_rl_impedance_hand(self):
        R, L, f = 12.0, 0.025, 60.0
        xl = TWO_PI * f * L
        zmag = R * xl / math.hypot(R, xl)
        c = Circuit.from_netlist([["I", "I1", "0", "a", [1, 0]], ["R", "R1", "a", "0", R], ["L", "L1", "a", "0", L]])
        self.assertAlmostEqual(abs(c.solve_ac(f).v("a")), zmag, places=12)

    def test_phasor_power(self):
        # 100 V rms at 30 degrees into 8 + j6 ohm: |I| = 10 A, P = 800 W, Q = 600 var
        c = Circuit()
        c.voltage_source("V1", "a", "0", C.polar(100, 30))
        c.resistor("R1", "a", "b", 8)
        w = 1000.0
        c.inductor("L1", "b", "0", 6 / w)
        s = c.solve(w)
        self.assertAlmostEqual(abs(s.i("R1")), 10.0, places=9)
        sv = s.power("V1")                                           # absorbed power: the source delivers, so negative
        self.assertAlmostEqual(sv.real, -800.0, places=6)
        self.assertAlmostEqual(sv.imag, -600.0, places=6)            # the inductive load takes +600 var from it
        self.assertAlmostEqual(s.power("R1").real, 800.0, places=6)

    def test_capacitor_island(self):
        c = Circuit()
        c.voltage_source("V1", "a", "0", 1)
        c.capacitor("C1", "a", "b", 1e-6)
        c.capacitor("C2", "b", "0", 1e-6)
        with self.assertRaises(SingularCircuitError):
            c.solve_dc()
        s = c.solve_ac(1000)
        self.assertAlmostEqual(abs(s.v("b")), 0.5, places=12)        # equal capacitors divide in half

    def test_dc_rejects_complex_source(self):
        c = Circuit()
        c.voltage_source("V1", "a", "0", 1 + 1j)
        c.resistor("R1", "a", "0", 1)
        with self.assertRaises(CircuitError):
            c.solve_dc()
        self.assertAlmostEqual(abs(c.solve_ac(10).i("R1")), math.sqrt(2), places=12)


class OpAmpTests(unittest.TestCase):
    def test_inverting(self):
        c = Circuit.from_netlist([["V", "V1", "in", "0", 0.35], ["R", "R1", "in", "m", 4700],
                                  ["R", "Rf", "m", "out", 47000], ["OA", "U1", "0", "m", "out"]])
        self.assertAlmostEqual(c.solve_dc().v("out"), C.inverting_amp(0.35, 4700, 47000), places=12)
        self.assertAlmostEqual(c.solve_dc().v("out"), -3.5, places=12)

    def test_non_inverting(self):
        c = Circuit.from_netlist([["V", "V1", "in", "0", 0.2], ["OA", "U1", "in", "m", "out"],
                                  ["R", "Rg", "m", "0", 2200], ["R", "Rf", "out", "m", 33000]])
        self.assertAlmostEqual(c.solve_dc().v("out"), C.non_inverting_amp(0.2, 2200, 33000), places=12)

    def test_summing_and_difference(self):
        vs, rs, rf = [0.8, 1.5, -0.2], [4000, 10000, 2000], 20000
        rows = [["OA", "U1", "0", "m", "out"], ["R", "Rf", "m", "out", rf]]
        for k, (v, r) in enumerate(zip(vs, rs)):
            rows += [["V", "V%d" % k, "s%d" % k, "0", v], ["R", "R%d" % k, "s%d" % k, "m", r]]
        out = Circuit.from_netlist(rows).solve_dc().v("out")
        self.assertAlmostEqual(out, C.summing_amp(vs, rs, rf), places=9)
        self.assertAlmostEqual(out, -5.0, places=9)
        # difference amplifier, unmatched resistors
        rows = [["V", "Va", "a", "0", 1.0], ["V", "Vb", "b", "0", 2.0],
                ["R", "R1", "a", "m", 10000], ["R", "R2", "m", "out", 40000],
                ["R", "R3", "b", "p", 20000], ["R", "R4", "p", "0", 30000],
                ["OA", "U1", "p", "m", "out"]]
        out = Circuit.from_netlist(rows).solve_dc().v("out")
        self.assertAlmostEqual(out, 2.0, places=9)
        self.assertAlmostEqual(out, C.difference_amp(1.0, 2.0, 10000, 40000, 20000, 30000), places=9)

    def test_output_current(self):
        c = Circuit.from_netlist([["V", "V1", "in", "0", 1], ["R", "R1", "in", "m", 1000],
                                  ["R", "Rf", "m", "out", 1000], ["R", "RL", "out", "0", 500],
                                  ["OA", "U1", "0", "m", "out"]], exact=True)
        s = c.solve_dc()
        self.assertEqual(s.v("out"), -1)
        # hand: Rf brings 1 mA into the output node and RL carries v/500 = -2 mA away from it
        # (2 mA actually flows ground -> out).  KCL: 1 mA + i_o = -2 mA, so the op-amp sinks 3 mA.
        self.assertEqual(s.i("U1"), F(-3, 1000))

    def test_open_loop(self):
        c = Circuit()
        c.voltage_source("V1", "in", "0", 1)
        c.resistor("R1", "in", "m", 1000)
        c.resistor("RL", "out", "0", 1000)
        c.opamp("U1", "0", "m", "out")             # no feedback: output undetermined
        with self.assertRaises(SingularCircuitError):
            c.solve_dc()

    def test_saturation_helpers(self):
        self.assertEqual(C.inverting_amp(1.5, 10000, 120000, v_sat=10.5), -10.5)
        self.assertEqual(C.non_inverting_amp(7.0, 1000, 1000, v_sat=12), 12)
        self.assertEqual(C.saturate(3, 5), 3)
        with self.assertRaises(ValueError):
            C.saturate(1, 1, 2)


class SingularAndBadInputTests(unittest.TestCase):
    def test_floating_resistor_island(self):
        c = Circuit()
        c.voltage_source("V1", "a", "0", 1)
        c.resistor("R1", "a", "0", 10)
        c.resistor("R2", "x", "y", 10)
        with self.assertRaises(SingularCircuitError) as cm:
            c.solve_dc()
        self.assertIn("floating", str(cm.exception))
        self.assertIn("x", str(cm.exception))

    def test_voltage_source_loop(self):
        c = Circuit()
        c.voltage_source("V1", "a", "0", 1)
        c.voltage_source("V2", "a", "0", 2)
        c.resistor("R1", "a", "0", 10)
        with self.assertRaises(SingularCircuitError):
            c.solve_dc()

    def test_current_source_cutset(self):
        c = Circuit()
        c.current_source("I1", "0", "a", 1)
        c.current_source("I2", "a", "b", 1)
        c.resistor("R1", "b", "0", 10)
        with self.assertRaises(SingularCircuitError):
            c.solve_dc()

    def test_no_ground_and_empty(self):
        c = Circuit()
        with self.assertRaises(CircuitError):
            c.solve_dc()
        c.resistor("R1", "a", "b", 10)
        with self.assertRaises(CircuitError):
            c.solve_dc()

    def test_bad_element_values(self):
        c = Circuit()
        for bad in (0, -5, float("nan"), float("inf"), True, "10", None, 1j):
            with self.assertRaises(CircuitError, msg=repr(bad)):
                c.resistor("R", "a", "0", bad)
        with self.assertRaises(CircuitError):
            c.capacitor("C", "a", "0", -1e-6)
        with self.assertRaises(CircuitError):
            c.inductor("L", "a", "0", 0)
        with self.assertRaises(CircuitError):
            c.voltage_source("V", "a", "0", float("nan"))
        with self.assertRaises(CircuitError):
            c.current_source("I", "a", "0", "1")

    def test_names_and_nodes(self):
        c = Circuit()
        c.resistor("R1", "a", "0", 1)
        with self.assertRaises(CircuitError):
            c.resistor("R1", "b", "0", 1)                           # duplicate name
        with self.assertRaises(CircuitError):
            c.resistor("R2", "a", "a", 1)                           # same node twice
        with self.assertRaises(CircuitError):
            c.resistor("", "a", "b", 1)
        with self.assertRaises(CircuitError):
            c.resistor(7, "a", "b", 1)
        c.voltage_source("V1", "a", "0", 1)
        s = c.solve_dc()
        with self.assertRaises(CircuitError):
            s.v("nope")
        with self.assertRaises(CircuitError):
            s.i("nope")
        with self.assertRaises(CircuitError):
            c.element("nope")
        self.assertEqual(Circuit().ground, "0")

    def test_solve_arguments(self):
        c = Circuit.from_netlist([["V", "V1", "a", "0", 1], ["R", "R1", "a", "0", 1]])
        for bad in (-1, float("nan"), "5", None):
            with self.assertRaises(CircuitError):
                c.solve(bad)
        for bad in (0, -50, float("inf"), "50"):
            with self.assertRaises(CircuitError):
                c.solve_ac(bad)

    def test_from_netlist_errors(self):
        for bad in (None, [], "R1", [["R", "R1", "a", "0"]], [["Q", "Q1", "a", "0", 1]],
                    [[1, "R1", "a", "0", 1]], [["R", "R1", "a", "0", 1, 2]],
                    [["V", "V1", "a", "0", [1, 2, 3]]], [["V", "V1", "a", "0", ["x", 0]]],
                    [["R", "R1", "a", "0", True]], [["R", "R1", "a", "0", float("nan")]]):
            with self.assertRaises(CircuitError, msg=repr(bad)):
                Circuit.from_netlist(bad, exact=True)

    def test_measure_parsing(self):
        s = Circuit.from_netlist([["V", "V1", "a", "0", 4], ["R", "R1", "a", "b", 2], ["R", "R2", "b", "0", 2]]).solve_dc()
        self.assertEqual(C.measure(s, "v(b)"), 2.0)
        self.assertEqual(C.measure(s, "v(a, b)"), 2.0)
        self.assertEqual(C.measure(s, "-i(R1)"), -1.0)
        self.assertEqual(C.measure(s, "p(R2)"), 2.0)
        self.assertEqual(C.measure(s, "v(a):abs"), 4.0)
        for bad in ("", "v()", "x(a)", "v(a,b,c)", "i(R1,R2)", "p(a,b)", "v(a):foo", None, 5, "v(nope)", "i(nope)"):
            with self.assertRaises(CircuitError, msg=repr(bad)):
                C.measure(s, bad)
        ac = Circuit.from_netlist([["V", "V1", "a", "0", 1], ["R", "R1", "a", "0", 1]]).solve_ac(50)
        with self.assertRaises(CircuitError):
            C.measure(ac, "v(a)")                                    # complex needs a part

    def test_op_amp_power_not_defined(self):
        c = Circuit.from_netlist([["V", "V1", "in", "0", 1], ["R", "R1", "in", "m", 1], ["R", "Rf", "m", "out", 1],
                                  ["OA", "U1", "0", "m", "out"]])
        with self.assertRaises(CircuitError):
            c.solve_dc().power("U1")


class TheveninNortonTests(unittest.TestCase):
    def test_thevenin_divider(self):
        c = Circuit.from_netlist([["V", "V1", "in", "0", 24], ["R", "R1", "in", "out", 4700],
                                  ["R", "R2", "out", "0", 10000]], exact=True)
        th = C.thevenin(c, "out")
        self.assertEqual(th.v_th, F(24 * 10000, 14700))
        self.assertEqual(th.z_th, F(4700 * 10000, 14700))
        self.assertAlmostEqual(float(th.v_th), C.voltage_divider(24, 4700, 10000), places=12)
        self.assertAlmostEqual(float(th.z_th), C.parallel(4700, 10000), places=9)

    def test_norton_consistency(self):
        c = Circuit.from_netlist([["V", "V1", "n1", "0", 18], ["R", "R1", "n1", "a", 3000], ["R", "R2", "a", "0", 6000],
                                  ["R", "R3", "a", "b", 2000], ["R", "R4", "b", "0", 4000]], exact=True)
        th = C.thevenin(c, "b")
        no = C.norton(c, "b")
        self.assertEqual(th.v_th, F(6))                              # by hand: 18 * 3/6 * 4/6 ... = 6 V
        self.assertEqual(th.z_th, F(2000))
        self.assertEqual(no.i_n, F(3, 1000))                         # = 6 V / 2 kohm
        self.assertEqual(th.i_n, no.i_n)
        self.assertEqual(no.z_n, th.z_th)

    def test_equivalent_load_current(self):
        rows = [["V", "V1", "n1", "0", 15], ["R", "R1", "n1", "a", 100], ["R", "R2", "a", "0", 150],
                ["R", "R3", "a", "n3", 30], ["I", "I1", "0", "n3", 0.04], ["R", "R5", "a", "b", 47],
                ["R", "R6", "b", "0", 220]]
        full = Circuit.from_netlist(rows + [["R", "RL", "b", "0", 330]], exact=True).solve_dc()
        th = C.thevenin(Circuit.from_netlist(rows, exact=True), "b")
        eq = Circuit()
        eq.voltage_source("Vth", "t", "0", th.v_th)
        eq.resistor("Rth", "t", "b", th.z_th)
        eq.resistor("RL", "b", "0", 330)
        self.assertEqual(eq.solve_dc().i("RL"), full.i("RL"))

    def test_deactivated_current_source(self):
        c = Circuit.from_netlist([["V", "V1", "n1", "0", 15], ["R", "R1", "n1", "a", 100], ["R", "R2", "a", "0", 150],
                                  ["R", "R3", "a", "n3", 30], ["I", "I1", "0", "n3", 0.04]], exact=True)
        self.assertEqual(C.thevenin(c, "a").z_th, F(60))

    def test_circuit_not_modified(self):
        c = Circuit.from_netlist([["V", "V1", "a", "0", 5], ["R", "R1", "a", "b", 10], ["R", "R2", "b", "0", 10]])
        before = c.elements
        C.thevenin(c, "b")
        C.norton(c, "b")
        self.assertEqual(c.elements, before)

    def test_port_without_ground_node(self):
        c = Circuit()
        c.resistor("R1", "A", "M", 10)
        c.resistor("R2", "M", "B", 20)
        self.assertEqual(C.thevenin(c, "A", "B").z_th, 30)

    def test_ac_thevenin(self):
        # source 10 V - R=30 - port a ; L (j40 at w) from a to ground
        w = 1000.0
        c = Circuit()
        c.voltage_source("V1", "s", "0", 10)
        c.resistor("R1", "s", "a", 30)
        c.inductor("L1", "a", "0", 0.04)
        th = C.thevenin(c.without("L1"), "a", freq=w / TWO_PI)
        self.assertAlmostEqual(abs(th.v_th - 10), 0, places=9)
        self.assertAlmostEqual(abs(th.z_th - 30), 0, places=9)

    def test_port_errors(self):
        c = Circuit.from_netlist([["V", "V1", "a", "0", 1], ["R", "R1", "a", "0", 1]])
        with self.assertRaises(CircuitError):
            C.thevenin(c, "zzz")
        with self.assertRaises(CircuitError):
            C.thevenin(c, "a", "a")
        with self.assertRaises(CircuitError):
            C.norton(c, "a")              # an ideal source across the port: no Norton short-circuit solution
        th = C.thevenin(c, "a")
        self.assertEqual(th.z_th, 0)
        with self.assertRaises(CircuitError):
            th.i_n


class FirstOrderTests(unittest.TestCase):
    def test_parameters_rc_divider(self):
        c = Circuit.from_netlist([["V", "V1", "in", "0", 20], ["R", "R1", "in", "x", 10000],
                                  ["R", "R2", "x", "0", 15000], ["C", "C1", "x", "0", 1e-5]], exact=True)
        v_final, tau = C.first_order_parameters(c, "C1")
        self.assertEqual(v_final, F(12))
        self.assertEqual(tau, F(6000) * F(1, 100000))
        self.assertAlmostEqual(C.first_order_response(c, "C1", 0, 0.05), 12 * (1 - math.exp(-0.05 / 0.06)), places=12)

    def test_rc_vs_simulation(self):
        # independent: integrate C dv/dt = (Vs - v)/R1 - v/R2 with the trapezoidal rule
        Vs, R1, R2, Cc, t_end = 20.0, 10000.0, 15000.0, 1e-5, 0.05
        c = Circuit.from_netlist([["V", "V1", "in", "0", Vs], ["R", "R1", "in", "x", R1],
                                  ["R", "R2", "x", "0", R2], ["C", "C1", "x", "0", Cc]])
        n = 20000
        dt = t_end / n
        a = -(1 / R1 + 1 / R2) / Cc
        b = Vs / (R1 * Cc)
        v = 0.0
        for _ in range(n):
            v = (v * (1 + a * dt / 2) + b * dt) / (1 - a * dt / 2)
        self.assertAlmostEqual(C.first_order_response(c, "C1", 0.0, t_end), v, places=6)

    def test_rl_network(self):
        # 12 V, R1=4 -a- R3=12 to gnd, R2=12 a-b, L=0.1 H b-gnd.
        # Vth = 9 V, Rth = 12 + 3 = 15 ohm -> I_final = 0.6 A, tau = 0.1/15 s
        c = Circuit.from_netlist([["V", "V1", "n1", "0", 12], ["R", "R1", "n1", "a", 4], ["R", "R3", "a", "0", 12],
                                  ["R", "R2", "a", "b", 12], ["L", "L1", "b", "0", F(1, 10)]], exact=True)
        i_final, tau = C.first_order_parameters(c, "L1")
        self.assertEqual(i_final, F(3, 5))
        self.assertEqual(tau, F(1, 150))
        self.assertAlmostEqual(C.first_order_response(c, "L1", 0.0, 0.01), 0.6 * (1 - math.exp(-1.5)), places=12)

    def test_one_storage_element_only(self):
        c = Circuit.from_netlist([["V", "V1", "a", "0", 1], ["R", "R1", "a", "b", 1], ["C", "C1", "b", "0", 1e-6],
                                  ["C", "C2", "b", "0", 1e-6]])
        with self.assertRaises(CircuitError):
            C.first_order_parameters(c, "C1")
        with self.assertRaises(CircuitError):
            C.first_order_parameters(c, "R1")

    def test_step_helpers(self):
        tau = C.time_constant_rc(47000, 22e-6)
        self.assertAlmostEqual(tau, 1.034, places=12)
        t = C.first_order_time(0, 9, tau, 6)
        self.assertAlmostEqual(C.first_order_step(0, 9, tau, t), 6.0, places=12)
        self.assertAlmostEqual(t, tau * math.log(3), places=12)
        self.assertAlmostEqual(C.time_to_fraction(0.01, 1 - math.exp(-1)), 0.01, places=12)   # 63.2 % at one tau
        self.assertAlmostEqual(C.time_constant_rl(0.25, 40), 0.00625, places=15)
        self.assertAlmostEqual(C.tau_from_points(8, 0, 2, 0.03), 0.03 / math.log(4), places=12)
        self.assertEqual(C.first_order_step(5, 1, 2, 0), 5)
        self.assertAlmostEqual(C.capacitor_energy(470e-6, 12), 0.03384, places=12)
        self.assertAlmostEqual(C.inductor_energy(0.05, 2), 0.1, places=12)

    def test_step_helper_errors(self):
        with self.assertRaises(ValueError):
            C.first_order_time(0, 9, 1.0, 10)          # beyond the final value
        with self.assertRaises(ValueError):
            C.first_order_time(0, 9, 1.0, -1)          # wrong side of the start
        with self.assertRaises(ValueError):
            C.first_order_time(5, 5, 1.0, 5)
        with self.assertRaises(ValueError):
            C.first_order_step(0, 1, 0, 1)
        with self.assertRaises(ValueError):
            C.first_order_step(0, 1, 1, -1)
        with self.assertRaises(ValueError):
            C.time_to_fraction(1, 1)
        with self.assertRaises(ValueError):
            C.tau_from_points(8, 0, 9, 1)
        with self.assertRaises(ValueError):
            C.time_constant_rc(-1, 1e-6)
        with self.assertRaises(ValueError):
            C.time_constant_rl(1e-3, float("nan"))


class ClosedFormTests(unittest.TestCase):
    def test_dividers(self):
        c = Circuit.from_netlist([["I", "I1", "0", "a", 0.015], ["R", "R1", "a", "0", 1000],
                                  ["R", "R2", "a", "0", 2200], ["R", "R3", "a", "0", 3300]], exact=True)
        s = c.solve_dc()
        self.assertAlmostEqual(float(s.i("R2")), C.current_divider(0.015, 2200, 1000, 3300), places=12)
        self.assertAlmostEqual(C.current_divider(1.0, 5), 1.0)
        self.assertEqual(C.voltage_divider(10, 1, 1), 5)
        with self.assertRaises(ValueError):
            C.voltage_divider(10, 0, 5)
        with self.assertRaises(ValueError):
            C.current_divider(1, 5, -1)

    def test_series_parallel(self):
        self.assertEqual(C.series(1, 2, 3), 6)
        self.assertAlmostEqual(C.parallel(60, 60), 30.0)
        self.assertEqual(C.parallel(10, 0), 0.0)
        with self.assertRaises(ValueError):
            C.parallel(1j, -1j)                         # ideal L and C in parallel at resonance: open circuit
        with self.assertRaises(ValueError):
            C.series()
        with self.assertRaises(ValueError):
            C.parallel()

    def test_max_power_transfer_scan(self):
        v, r = 20.0, 8.0
        rl, pmax = C.max_power_transfer(v, r)
        self.assertEqual((rl, pmax), (8.0, 12.5))
        best = max((C.power_in_load(v, r, x / 100.0), x / 100.0) for x in range(10, 10000))
        self.assertAlmostEqual(best[1], rl, places=2)
        self.assertAlmostEqual(best[0], pmax, places=6)
        self.assertAlmostEqual(C.power_in_load(20, 8, 12), 12.0, places=12)

    def test_max_power_ac_scan(self):
        vrms, zth = 120.0, 30 + 40j
        zl, p = C.max_power_transfer_ac(vrms, zth, "complex")
        self.assertEqual(zl, 30 - 40j)
        self.assertAlmostEqual(p, 120.0, places=9)
        # brute force over resistive loads
        def power(rl):
            i = vrms / abs(zth + rl)
            return i * i * rl
        best = max((power(x / 100.0), x / 100.0) for x in range(100, 20000))
        rl, p2 = C.max_power_transfer_ac(vrms, zth, "resistive")
        self.assertAlmostEqual(best[1], rl, places=1)
        self.assertAlmostEqual(best[0], p2, places=5)
        self.assertAlmostEqual(p2, 90.0, places=9)
        # complex scan: no (R, X) pair beats the conjugate match
        self.assertTrue(all(vrms ** 2 * r / abs(zth + complex(r, x)) ** 2 <= p + 1e-9
                            for r in (5, 20, 30, 60) for x in (-80, -40, 0, 40)))
        with self.assertRaises(ValueError):
            C.max_power_transfer_ac(1, 5j)
        with self.assertRaises(ValueError):
            C.max_power_transfer_ac(1, 5 + 1j, "weird")

    def test_series_band_edges(self):
        R, L, Cc = 8.0, 2e-3, 0.5e-6
        res = C.series_rlc_resonance(R, L, Cc)
        self.assertAlmostEqual(res.f0, 5032.921210448704, places=6)
        self.assertAlmostEqual(res.q, math.sqrt(L / Cc) / R, places=12)
        self.assertAlmostEqual(res.bandwidth, R / (TWO_PI * L), places=9)
        def current(f):
            ckt = Circuit()
            ckt.voltage_source("V1", "in", "0", 1)
            ckt.resistor("R1", "in", "a", R)
            ckt.inductor("L1", "a", "b", L)
            ckt.capacitor("C1", "b", "0", Cc)
            return abs(ckt.solve_ac(f).i("R1"))
        q = res.q
        f1 = res.f0 * (math.sqrt(1 + 1 / (4 * q * q)) - 1 / (2 * q))
        f2 = res.f0 * (math.sqrt(1 + 1 / (4 * q * q)) + 1 / (2 * q))
        self.assertAlmostEqual(current(f1) / current(res.f0), 1 / math.sqrt(2), places=9)
        self.assertAlmostEqual(current(f2) / current(res.f0), 1 / math.sqrt(2), places=9)
        self.assertAlmostEqual(f2 - f1, res.bandwidth, places=6)

    def test_parallel_band_edges(self):
        R, L, Cc = 5000.0, 100e-6, 2e-9
        res = C.parallel_rlc_resonance(R, L, Cc)
        self.assertAlmostEqual(res.q, R * math.sqrt(Cc / L), places=12)
        self.assertAlmostEqual(res.bandwidth, 1 / (TWO_PI * R * Cc), places=3)
        def vmag(f):
            ckt = Circuit()
            ckt.current_source("I1", "0", "a", 1)
            ckt.resistor("R1", "a", "0", R)
            ckt.inductor("L1", "a", "0", L)
            ckt.capacitor("C1", "a", "0", Cc)
            return abs(ckt.solve_ac(f).v("a"))
        q = res.q
        f1 = res.f0 * (math.sqrt(1 + 1 / (4 * q * q)) - 1 / (2 * q))
        f2 = res.f0 * (math.sqrt(1 + 1 / (4 * q * q)) + 1 / (2 * q))
        self.assertAlmostEqual(vmag(res.f0), R, places=4)
        self.assertAlmostEqual(vmag(f1) / R, 1 / math.sqrt(2), places=9)
        self.assertAlmostEqual(vmag(f2) / R, 1 / math.sqrt(2), places=9)
        self.assertAlmostEqual(f2 - f1, res.bandwidth, places=3)

    def test_resonant_capacitance_and_corner(self):
        c = C.resonant_capacitance(1e6, 180e-6)
        self.assertAlmostEqual(C.series_rlc_resonance(1.0, 180e-6, c).f0, 1e6, places=3)
        self.assertAlmostEqual(C.rc_corner_frequency(1000, 1e-7), 1591.549430918953, places=9)
        self.assertAlmostEqual(C.series_rlc_impedance(22, 15e-3, 33e-6, 50).imag, 2 * math.pi * 50 * 15e-3 - 1 / (2 * math.pi * 50 * 33e-6), places=9)
        with self.assertRaises(ValueError):
            C.series_rlc_resonance(0, 1e-3, 1e-6)
        with self.assertRaises(ValueError):
            C.impedance_inductor(1e-3, 0)

    def test_rms(self):
        dc, peaks = 5.0, [10.0, 3.0]
        n = 200000
        total = 0.0
        for k in range(n):
            th = TWO_PI * (k + 0.5) / n
            v = dc + peaks[0] * math.sin(th) + peaks[1] * math.sin(3 * th)
            total += v * v
        self.assertAlmostEqual(math.sqrt(total / n), C.rms_composite(dc, peaks), places=6)
        self.assertAlmostEqual(C.rms_sine(311.0), 311 / math.sqrt(2), places=12)
        self.assertAlmostEqual(C.METHODS["avg_power_sine_peak"](170, 15), (170 / math.sqrt(2)) ** 2 / 15, places=9)

    def test_pf_correction(self):
        p, v, f, pf1, pf2 = 3000.0, 230.0, 50.0, 0.75, 0.95
        c = C.pf_correction_capacitance(p, v, f, pf1, pf2)
        s_load = complex(p, p * math.tan(math.acos(pf1)))           # lagging: +jQ
        s_cap = -1j * TWO_PI * f * c * v * v                          # capacitor supplies reactive power
        s_total = s_load + s_cap
        self.assertAlmostEqual(s_total.real / abs(s_total), pf2, places=12)
        self.assertAlmostEqual(c * 1e6, 99.87, places=1)
        self.assertAlmostEqual(C.reactive_power(4000, 0.8), 3000.0, places=9)
        self.assertAlmostEqual(C.apparent_power(4000, 0.8), 5000.0, places=9)
        self.assertEqual(C.pf_correction_reactive(1000, 0.9, 0.9), 0)
        with self.assertRaises(ValueError):
            C.pf_correction_reactive(1000, 0.9, 0.8)
        for bad in (0, 1.2, -0.5, float("nan")):
            with self.assertRaises(ValueError):
                C.reactive_power(100, bad)

    def test_pf_correction_delta_wye(self):
        cd = C.pf_correction_capacitance(90e3, 400, 50, 0.8, 0.95, phases=3, connection="delta")
        cw = C.pf_correction_capacitance(90e3, 400, 50, 0.8, 0.95, phases=3, connection="wye")
        self.assertAlmostEqual(cw / cd, 3.0, places=12)
        qc = 90e3 * (0.75 - math.tan(math.acos(0.95)))
        # delta: each capacitor across 400 V supplies qc/3
        self.assertAlmostEqual(3 * TWO_PI * 50 * cd * 400 ** 2, qc, places=6)
        with self.assertRaises(ValueError):
            C.pf_correction_capacitance(1, 400, 50, 0.8, 0.9, phases=2)
        with self.assertRaises(ValueError):
            C.pf_correction_capacitance(1, 400, 50, 0.8, 0.9, phases=3, connection="star")

    def test_balanced_three_phase(self):
        vl, z = 400.0, 30 + 40j
        w = C.balanced_load("wye", vl, z)
        va = vl / math.sqrt(3)
        i = va / z
        self.assertAlmostEqual(w.i_line, abs(i), places=12)
        self.assertAlmostEqual(w.p, 3 * (abs(i) ** 2) * z.real, places=9)
        self.assertAlmostEqual(w.q, 3 * (abs(i) ** 2) * z.imag, places=9)
        self.assertAlmostEqual(w.s, math.hypot(w.p, w.q), places=9)
        d = C.balanced_load("delta", vl, z)
        iph = vl / z
        self.assertAlmostEqual(d.i_phase, abs(iph), places=12)
        self.assertAlmostEqual(d.i_line, math.sqrt(3) * abs(iph), places=12)
        self.assertAlmostEqual(d.p, 3 * 3 * (abs(i) ** 2) * z.real, places=9)    # delta draws 3x the power of wye
        self.assertAlmostEqual(d.p, 5760.0, places=9)
        p, q, s = C.three_phase_power(415, 15, 0.85)
        self.assertAlmostEqual(p, math.sqrt(3) * 415 * 15 * 0.85, places=9)
        self.assertAlmostEqual(math.hypot(p, q), s, places=9)
        self.assertAlmostEqual(C.line_current(55e3, 415, 0.85, phases=3), 55e3 / (math.sqrt(3) * 415 * 0.85), places=9)
        self.assertAlmostEqual(C.line_current(3000, 230, 0.75), 3000 / (230 * 0.75), places=9)
        with self.assertRaises(ValueError):
            C.balanced_load("star", 400, 10)
        with self.assertRaises(ValueError):
            C.balanced_load("wye", 400, 0)
        with self.assertRaises(ValueError):
            C.line_current(1, 1, 1, phases=2)

    def test_delta_wye(self):
        za, zb, zc = 12.0, 30.0, 47.0                                 # an unbalanced delta (A-B, B-C, C-A)
        y = C.delta_to_wye(za, zb, zc)
        back = C.wye_to_delta(*y)
        for a, b in zip(back, (za, zb, zc)):
            self.assertAlmostEqual(a, b, places=9)
        self.assertEqual(C.delta_to_wye(60, 60, 60), (20, 20, 20))
        self.assertEqual(C.wye_to_delta(20, 20, 20), (60, 60, 60))
        delta = Circuit()
        delta.resistor("Rab", "A", "B", za); delta.resistor("Rbc", "B", "C", zb); delta.resistor("Rca", "C", "A", zc)
        wye = Circuit()
        wye.resistor("Ra", "A", "O", y[0]); wye.resistor("Rb", "B", "O", y[1]); wye.resistor("Rc", "C", "O", y[2])
        for p, q in (("A", "B"), ("B", "C"), ("C", "A")):
            self.assertAlmostEqual(C.thevenin(delta, p, q).z_th, C.thevenin(wye, p, q).z_th, places=9)

    def test_per_unit(self):
        zb = C.base_impedance(100e6, 66e3)
        self.assertAlmostEqual(zb, 43.56, places=9)
        self.assertAlmostEqual(C.to_per_unit(12, zb), 0.27548209366391185, places=12)
        self.assertAlmostEqual(C.from_per_unit(C.to_per_unit(12, zb), zb), 12.0, places=12)
        self.assertAlmostEqual(C.base_current(100e6, 132e3), 100e6 / (math.sqrt(3) * 132e3), places=9)
        self.assertAlmostEqual(C.base_current(1000, 100, phases=1), 10.0)
        # a base change must give the same answer as converting the ohmic value directly
        s_old, v_old, s_new, v_new, z_pu = 20e6, 13.2e3, 100e6, 13.8e3, 0.18
        ohms = z_pu * C.base_impedance(s_old, v_old)
        direct = ohms / C.base_impedance(s_new, v_new)
        self.assertAlmostEqual(C.change_base(z_pu, s_old, v_old, s_new, v_new), direct, places=12)
        self.assertAlmostEqual(direct, 0.8234, places=3)
        self.assertAlmostEqual(C.METHODS["fault_current_series"](50e6, 33e3, [0.15, 0.10]),
                               4.0 * 50e6 / (math.sqrt(3) * 33e3), places=6)
        with self.assertRaises(ValueError):
            C.base_impedance(0, 1)
        with self.assertRaises(ValueError):
            C.base_current(1, 1, phases=2)

    def test_difference_amp_matched(self):
        self.assertAlmostEqual(C.difference_amp(1.0, 1.2, 10e3, 47e3, 10e3, 47e3), 4.7 * 0.2, places=12)
        with self.assertRaises(ValueError):
            C.summing_amp([1], [1, 2], 1)
        with self.assertRaises(ValueError):
            C.inverting_amp(1, 0, 1)


class RegistryTests(unittest.TestCase):
    def test_get_method(self):
        self.assertIs(C.get_method("netlist_measure"), C.METHODS["netlist_measure"])
        self.assertIs(C.get_method("voltage_divider"), C.voltage_divider)
        for bad in ("nope", "Circuit", "__import__", "", "os", "METHODS"):
            with self.assertRaises(KeyError, msg=bad):
                C.get_method(bad)

    def test_wrappers(self):
        m = C.METHODS
        net = [["V", "V1", "in", "0", 24], ["R", "R1", "in", "out", 4700], ["R", "R2", "out", "0", 10000],
               ["R", "RL", "out", "0", 22000]]
        rp = 10000 * 22000 / 32000
        self.assertAlmostEqual(m["netlist_measure"](net, "v(out)"), 24 * rp / (4700 + rp), places=9)
        self.assertAlmostEqual(m["load_power_netlist"](
            [["V", "V1", "a", "0", 12], ["R", "R1", "a", "b", 4]], "b", "0", 8), 8 * (12 / 12) ** 2, places=12)
        self.assertAlmostEqual(m["resonance"]("series", 8, 2e-3, 0.5e-6, "q"), 7.905694150420948, places=9)
        self.assertAlmostEqual(m["three_phase_load_voltage"](400, 0.0, 0.0, 10, 0, "wye"), 400.0, places=9)
        with self.assertRaises(ValueError):
            m["resonance"]("diagonal", 1, 1, 1, "f0")
        with self.assertRaises(ValueError):
            m["resonance"]("series", 1, 1, 1, "bogus")
        with self.assertRaises(ValueError):
            m["balanced_load_output"]("wye", 400, 3, 4, "bogus")

    def test_three_phase_feeder(self):
        zl, zd = complex(0.5, 1.0), complex(45, 24)
        zy = zd / 3
        v = (400 / math.sqrt(3)) * zy / (zl + zy)
        self.assertAlmostEqual(C.METHODS["three_phase_load_voltage"](400, 0.5, 1.0, 45, 24, "delta"),
                               math.sqrt(3) * abs(v), places=9)
        # capacitive load and purely reactive line also work
        got = C.METHODS["three_phase_load_voltage"](400, 0.0, 2.0, 20, -15, "wye")
        zl2, z2 = 2j, 20 - 15j
        self.assertAlmostEqual(got, math.sqrt(3) * abs((400 / math.sqrt(3)) * z2 / (zl2 + z2)), places=9)


class MiscTests(unittest.TestCase):
    def test_voltage_follower(self):
        c = Circuit.from_netlist([["V", "V1", "in", "0", F(33, 10)], ["OA", "U1", "in", "out", "out"],
                                  ["R", "RL", "out", "0", 100]], exact=True)
        s = c.solve_dc()
        self.assertEqual(s.v("out"), F(33, 10))
        self.assertEqual(s.i("U1"), F(33, 1000))                   # the op-amp supplies the load current
        with self.assertRaises(CircuitError):
            Circuit().opamp("U", "a", "a", "b")

    def test_values_too_big_for_float(self):
        c = Circuit()
        for bad in (10 ** 400, F(1, 10 ** 400), 5e-324 * 0, float("inf")):
            with self.assertRaises(CircuitError, msg=repr(bad)):
                c.resistor("R", "a", "0", bad)
        with self.assertRaises(CircuitError):
            c.voltage_source("V", "a", "0", 10 ** 400)
        with self.assertRaises(CircuitError):
            c.voltage_source("V", "a", "0", complex(1, float("nan")))

    def test_random_circuits(self):
        import random
        rng = random.Random(2024)

        def build(ac):
            n = rng.randint(2, 6)
            nodes = ["0"] + ["n%d" % i for i in range(1, n)]
            c = Circuit()
            for i in range(1, n):                                  # a resistor tree keeps every node grounded
                c.resistor("Rt%d" % i, nodes[i], nodes[rng.randint(0, i - 1)], rng.choice([10, 47, 100, 1000, 4700]))
            for k in range(rng.randint(1, 6)):
                a, b = rng.sample(nodes, 2)
                kind = rng.choice(["R", "R", "V", "I"] + (["L", "C"] if ac else []))
                if kind == "R":
                    c.resistor("R%d" % k, a, b, rng.choice([1, 22, 330, 10000]))
                elif kind == "L":
                    c.inductor("L%d" % k, a, b, rng.choice([1e-3, 0.1]))
                elif kind == "C":
                    c.capacitor("C%d" % k, a, b, rng.choice([1e-7, 1e-5]))
                elif kind == "V":
                    c.voltage_source("V%d" % k, a, b, complex(rng.randint(1, 20), rng.randint(-5, 5)) if ac else rng.randint(1, 20))
                else:
                    c.current_source("I%d" % k, a, b, complex(rng.random(), rng.random()) if ac else round(rng.random(), 3))
            return c

        solved = 0
        for trial in range(300):
            ac = trial % 2 == 0
            c = build(ac)
            omega = 2 * math.pi * rng.choice([50, 1000, 12345]) if ac else 0
            try:
                s = c.solve(omega)
            except SingularCircuitError:
                continue                                          # e.g. two voltage sources in parallel
            solved += 1
            scale = max(1.0, max(abs(s.i(el.name)) for el in c.elements))
            for node in c.nodes():
                total = 0
                for el in c.elements:
                    if node == el.nodes[0]:
                        total += s.i(el.name)
                    elif node == el.nodes[1]:
                        total -= s.i(el.name)
                self.assertLess(abs(total) / scale, 1e-9)
            for el in c.elements:
                v = s.v(el.nodes[0], el.nodes[1])
                if el.kind == "R":
                    self.assertLess(abs(v - s.i(el.name) * el.value), 1e-8 * max(1, abs(v)))
                elif el.kind == "V":
                    self.assertLess(abs(v - el.value), 1e-8 * max(1, abs(v)))
                elif el.kind == "L":
                    self.assertLess(abs(v - 1j * omega * el.value * s.i(el.name)), 1e-8 * max(1, abs(v)))
                elif el.kind == "C":
                    self.assertLess(abs(s.i(el.name) - 1j * omega * el.value * v), 1e-8 * max(1, abs(s.i(el.name))))
        self.assertGreater(solved, 200)

    def test_wide_resistor_range(self):
        c = Circuit()
        c.voltage_source("V1", "a", "0", 5)
        c.resistor("R1", "a", "b", 1e-3)
        c.resistor("R2", "b", "c", 1e9)
        c.resistor("R3", "c", "0", 1e3)
        s = c.solve_dc()
        self.assertAlmostEqual(s.v("b"), 5.0, places=6)
        self.assertAlmostEqual(s.v("c"), 5.0 * 1e3 / (1e9 + 1e3), places=12)


if __name__ == "__main__":
    unittest.main()
