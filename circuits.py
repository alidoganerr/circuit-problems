"""Circuit solver and formula helpers used by the problem bank.

Circuit does modified nodal analysis on a netlist of R, L, C, V and I sources
and ideal op-amps, either at DC (L shorted, C open) or in AC steady state at one
frequency. If every value in a DC circuit is an int or Fraction the answer comes
back as an exact Fraction, otherwise as float; AC answers are complex. The rest of the module is the
usual textbook formulas, plus METHODS, the wrappers problems.jsonl refers to.

Conventions:
- SI units. Angles in radians unless the name ends in _deg.
- An element's current flows from its first node to its second node through
  the element. A voltage source V(name, p, n, E) has v(p) - v(n) = E, so a
  source that is delivering power has a negative current. A current source
  I(name, f, t, value) pushes value amperes from f to t through itself.
- Phasors come out on whatever scale the sources use. RMS sources give RMS
  results and power in W/var/VA; peak sources give peak results and a power
  that is twice the average.
- One frequency per solve.
"""

import cmath
import math
import re
from collections import namedtuple
from fractions import Fraction
from numbers import Real

__all__ = [
    "CircuitError", "SingularCircuitError", "Circuit", "Solution", "Element",
    "Thevenin", "Norton", "thevenin", "norton", "measure",
    "first_order_parameters", "first_order_response",
    "series", "parallel", "polar", "to_polar",
    "voltage_divider", "current_divider",
    "power_in_load", "max_power_transfer", "max_power_transfer_ac",
    "time_constant_rc", "time_constant_rl", "first_order_step",
    "first_order_time", "time_to_fraction", "tau_from_points",
    "capacitor_energy", "inductor_energy",
    "impedance_inductor", "impedance_capacitor", "series_rlc_impedance",
    "rms_sine", "rms_composite",
    "rc_corner_frequency", "Resonance", "series_rlc_resonance",
    "parallel_rlc_resonance", "resonant_capacitance",
    "reactive_power", "apparent_power", "line_current",
    "pf_correction_reactive", "pf_correction_capacitance",
    "ThreePhase", "balanced_load", "three_phase_power",
    "delta_to_wye", "wye_to_delta",
    "base_impedance", "base_current", "to_per_unit", "from_per_unit",
    "change_base",
    "inverting_amp", "non_inverting_amp", "summing_amp", "difference_amp",
    "saturate",
    "METHODS", "get_method",
]

SQRT3 = math.sqrt(3.0)


class CircuitError(ValueError):
    """Bad circuit description or an unsolvable request."""


class SingularCircuitError(CircuitError):
    """The circuit has no unique solution (floating node, source loop, ...)."""


def _is_real(x):
    return isinstance(x, (Real, Fraction)) and not isinstance(x, bool)


def _real_value(x, what, positive=False):
    # Fractions and big ints can be finite yet overflow (or underflow to 0.0)
    # once converted to float, so check the float as well.
    if not _is_real(x):
        raise CircuitError("%s must be a real number, got %r" % (what, x))
    try:
        f = float(x)
    except OverflowError:
        raise CircuitError("%s is too large for a float" % what)
    if not math.isfinite(f):
        raise CircuitError("%s must be finite, got %r" % (what, x))
    if positive and not (x > 0 and f > 0):
        raise CircuitError("%s must be positive (and not underflow to zero), got %r" % (what, x))
    return x


def _source_value(x, what):
    if isinstance(x, complex):
        if not (math.isfinite(x.real) and math.isfinite(x.imag)):
            raise CircuitError("%s must be finite, got %r" % (what, x))
        return x
    return _real_value(x, what)


# The closed-form helpers raise ValueError, the circuit code CircuitError.
def _positive(x, what):
    if not _is_real(x) or not math.isfinite(float(x)) or not x > 0:
        raise ValueError("%s must be a positive finite number, got %r" % (what, x))
    return float(x)


def _finite(x, what):
    if isinstance(x, complex) or not _is_real(x) or not math.isfinite(float(x)):
        raise ValueError("%s must be a finite real number, got %r" % (what, x))
    return float(x)


def _pf(x, what="power factor"):
    x = _finite(x, what)
    if not 0 < x <= 1:
        raise ValueError("%s must be in (0, 1], got %r" % (what, x))
    return x


_SINGULAR_TOL = 1e-12


def _solve_linear(A, b, exact):
    """Solve A x = b by Gaussian elimination.

    Exact (Fraction) arithmetic picks any non-zero pivot.  Floating/complex
    arithmetic scales each row to unit max-norm and uses partial pivoting; a
    pivot below ``_SINGULAR_TOL`` means the system is singular.
    """
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    if not exact:
        for i in range(n):
            s = max(abs(x) for x in M[i][:n])
            if s > 0:
                M[i] = [x / s for x in M[i]]
    for col in range(n):
        if exact:
            piv = next((r for r in range(col, n) if M[r][col] != 0), None)
        else:
            piv = max(range(col, n), key=lambda r: abs(M[r][col]))
            if abs(M[piv][col]) <= _SINGULAR_TOL:
                piv = None
        if piv is None:
            raise SingularCircuitError(
                "singular circuit: no unique solution (check for a loop of "
                "voltage sources/inductors, a cut-set of current sources, "
                "an unconstrained op-amp output, a floating sub-circuit, or element "
                "values spanning an absurd range)")
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for r in range(col + 1, n):
            f = M[r][col]
            if f != 0:
                f = f / pv
                M[r] = [x - f * y for x, y in zip(M[r], M[col])]
    x = [None] * n
    for r in range(n - 1, -1, -1):
        s = M[r][n]
        for c in range(r + 1, n):
            s -= M[r][c] * x[c]
        x[r] = s / M[r][r]
    if not exact and not all(math.isfinite(abs(v)) for v in x):
        raise SingularCircuitError("singular circuit: solution is not finite")
    return x


Element = namedtuple("Element", "kind name nodes value")

_KINDS = ("R", "L", "C", "V", "I", "OA")


class Solution:
    """What Circuit.solve returns.

    v(node, ref) is v(node) - v(ref), ref defaulting to ground. i(name) is the
    element current, first node to second; for an op-amp it is the current
    the output pushes into its node. power(name) is absorbed power (V*conj(I)
    for AC), so it is negative for an element that delivers power.
    """

    def __init__(self, circuit, omega, voltages, currents):
        self._circuit = circuit
        self.omega = omega
        self.voltages = voltages
        self.currents = currents

    def v(self, node, ref=None):
        node = str(node)
        ref = self._circuit.ground if ref is None else str(ref)
        for n in (node, ref):
            if n not in self.voltages:
                raise CircuitError("unknown node %r" % n)
        return self.voltages[node] - self.voltages[ref]

    def i(self, name):
        if name not in self.currents:
            raise CircuitError("unknown element %r" % (name,))
        return self.currents[name]

    def power(self, name):
        el = self._circuit.element(name)
        if el.kind == "OA":
            raise CircuitError("power is not defined for an ideal op-amp")
        a, b = el.nodes
        v = self.v(a, b)
        i = self.i(name)
        return v * i if self.omega == 0 else v * i.conjugate()


class Circuit:
    """Add elements with resistor(), voltage_source() etc., then call
    solve_dc() or solve_ac(f). Node names are converted to str; ground is "0"
    unless you say otherwise.
    """

    def __init__(self, ground="0"):
        self.ground = str(ground)
        self._elements = []
        self._by_name = {}

    def _add(self, kind, name, nodes, value):
        if not isinstance(name, str) or not name.strip():
            raise CircuitError("element name must be a non-empty string, got %r" % (name,))
        if name in self._by_name:
            raise CircuitError("duplicate element name %r" % name)
        nodes = tuple(str(n) for n in nodes)
        if kind == "OA":
            # the output may share a node with an input (voltage follower, positive
            # feedback), but the two inputs must be different nodes
            if nodes[0] == nodes[1]:
                raise CircuitError("op-amp %r has both inputs on the same node" % name)
        elif len(set(nodes)) != len(nodes):
            raise CircuitError("element %r has two terminals on the same node" % name)
        el = Element(kind, name, nodes, value)
        self._elements.append(el)
        self._by_name[name] = el
        return el

    def resistor(self, name, a, b, ohms):
        _real_value(ohms, "resistance of %s" % name, positive=True)
        return self._add("R", name, (a, b), ohms)

    def inductor(self, name, a, b, henries):
        _real_value(henries, "inductance of %s" % name, positive=True)
        return self._add("L", name, (a, b), henries)

    def capacitor(self, name, a, b, farads):
        _real_value(farads, "capacitance of %s" % name, positive=True)
        return self._add("C", name, (a, b), farads)

    def voltage_source(self, name, plus, minus, volts):
        # volts may be complex (a phasor) for AC
        _source_value(volts, "value of %s" % name)
        return self._add("V", name, (plus, minus), volts)

    def current_source(self, name, from_node, to_node, amps):
        _source_value(amps, "value of %s" % name)
        return self._add("I", name, (from_node, to_node), amps)

    def opamp(self, name, non_inverting, inverting, output):
        """Ideal op-amp: v(+) = v(-), no input current, no output limit.
        The output may share a node with an input (voltage follower)."""
        return self._add("OA", name, (non_inverting, inverting, output), None)

    @property
    def elements(self):
        return tuple(self._elements)

    def element(self, name):
        try:
            return self._by_name[name]
        except (KeyError, TypeError):
            raise CircuitError("unknown element %r" % (name,))

    def nodes(self):
        seen = []
        for el in self._elements:
            for n in el.nodes:
                if n not in seen:
                    seen.append(n)
        return seen

    def copy(self, ground=None):
        c = Circuit(self.ground if ground is None else ground)
        for el in self._elements:
            c._add(el.kind, el.name, el.nodes, el.value)
        return c

    def without(self, name):
        self.element(name)              # raises if the name is unknown
        c = Circuit(self.ground)
        for el in self._elements:
            if el.name != name:
                c._add(el.kind, el.name, el.nodes, el.value)
        return c

    def deactivated(self):
        # every independent source zeroed: V becomes a short, I an open
        c = Circuit(self.ground)
        for el in self._elements:
            value = 0 if el.kind in ("V", "I") else el.value
            c._add(el.kind, el.name, el.nodes, value)
        return c

    @classmethod
    def from_netlist(cls, rows, ground="0", exact=False):
        """Build from rows like ["R", "R1", "a", "b", 1000] or
        ["OA", "U1", non_inv, inv, out]. A V or I value may be a phasor
        [magnitude, angle_deg]. With exact=True numbers go through Fraction(repr(x)),
        so 0.1 is exactly 1/10 and DC answers stay exact.
        """
        if not isinstance(rows, (list, tuple)) or not rows:
            raise CircuitError("netlist must be a non-empty list of rows")
        c = cls(ground)
        for row in rows:
            if not isinstance(row, (list, tuple)) or not row or not isinstance(row[0], str):
                raise CircuitError("bad netlist row %r" % (row,))
            kind = row[0].upper()
            if kind not in _KINDS:
                raise CircuitError("unknown element kind %r in row %r" % (row[0], row))
            if len(row) != 5:
                raise CircuitError("row %r should have 5 fields" % (row,))
            name = row[1]
            if kind == "OA":
                c.opamp(name, row[2], row[3], row[4])
                continue
            a, b, val = row[2], row[3], row[4]
            if kind in ("V", "I") and isinstance(val, (list, tuple)):
                if len(val) != 2 or not all(_is_real(x) for x in val):
                    raise CircuitError("phasor value must be [magnitude, angle_deg]: %r" % (val,))
                val = polar(float(val[0]), float(val[1]))
            elif exact and _is_real(val) and not isinstance(val, Fraction):
                if isinstance(val, float) and not math.isfinite(val):
                    raise CircuitError("value of %s must be finite" % name)
                val = Fraction(repr(val))
            if kind == "R":
                c.resistor(name, a, b, val)
            elif kind == "L":
                c.inductor(name, a, b, val)
            elif kind == "C":
                c.capacitor(name, a, b, val)
            elif kind == "V":
                c.voltage_source(name, a, b, val)
            else:
                c.current_source(name, a, b, val)
        return c

    def _floating_nodes(self, omega):
        # union-find over the elements that tie two node voltages together;
        # capacitors only count for AC
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        for n in self.nodes():
            find(n)
        for el in self._elements:
            if el.kind in ("R", "L", "V"):
                union(el.nodes[0], el.nodes[1])
            elif el.kind == "C" and omega > 0:
                union(el.nodes[0], el.nodes[1])
            elif el.kind == "OA":
                union(el.nodes[0], el.nodes[1])      # v+ = v-
        g = find(self.ground)
        return sorted(n for n in self.nodes() if find(n) != g)

    def solve_dc(self):
        return self.solve(0)        # C open, L short

    def solve_ac(self, freq_hz):
        if not _is_real(freq_hz) or not math.isfinite(float(freq_hz)) or not freq_hz > 0:
            raise CircuitError("frequency must be a positive finite number, got %r" % (freq_hz,))
        return self.solve(2.0 * math.pi * float(freq_hz))

    def solve(self, omega=0):
        """Solve at omega rad/s (0 is DC)."""
        if not _is_real(omega) or not math.isfinite(float(omega)) or omega < 0:
            raise CircuitError("omega must be a finite number >= 0, got %r" % (omega,))
        if not self._elements:
            raise CircuitError("the circuit has no elements")
        if self.ground not in self.nodes():
            raise CircuitError("no element is connected to the ground node %r" % self.ground)
        floating = self._floating_nodes(omega)
        if floating:
            raise SingularCircuitError(
                "floating node(s) with no path to ground: %s" % ", ".join(floating))

        dc = (omega == 0)
        values = [el.value for el in self._elements if el.value is not None]
        if dc and any(isinstance(v, complex) for v in values):
            raise CircuitError("DC solve needs real source values; use solve_ac for phasors")
        if not dc or any(isinstance(v, complex) for v in values):
            mode = "complex"
        elif all(isinstance(v, (int, Fraction)) for v in values):
            mode = "exact"
        else:
            mode = "float"
        if mode == "exact":
            cast, zero, one = Fraction, Fraction(0), Fraction(1)
        elif mode == "float":
            cast, zero, one = float, 0.0, 1.0
        else:
            cast, zero, one = complex, 0j, 1 + 0j
        w = cast(omega)

        nodes = [n for n in self.nodes() if n != self.ground]
        idx = {n: k for k, n in enumerate(nodes)}
        N = len(nodes)
        branch = [el for el in self._elements if el.kind in ("V", "L", "OA")]
        bidx = {el.name: N + k for k, el in enumerate(branch)}
        size = N + len(branch)
        A = [[zero] * size for _ in range(size)]
        rhs = [zero] * size

        def add(r, c, val):
            # ground has no row/column; ix() returns None for it
            if r is not None and c is not None:
                A[r][c] += val

        ix = idx.get

        for el in self._elements:
            if el.kind in ("R", "C"):
                a, b = ix(el.nodes[0]), ix(el.nodes[1])
                if el.kind == "R":
                    y = one / cast(el.value)
                elif dc:
                    continue
                else:
                    y = 1j * w * cast(el.value)
                add(a, a, y); add(b, b, y); add(a, b, -y); add(b, a, -y)
            elif el.kind in ("V", "L"):
                p, n = ix(el.nodes[0]), ix(el.nodes[1])
                k = bidx[el.name]
                add(p, k, one); add(n, k, -one)      # KCL: branch current leaves p, enters n
                add(k, p, one); add(k, n, -one)      # v_p - v_n = ...
                if el.kind == "V":
                    rhs[k] = cast(el.value)
                else:
                    A[k][k] -= (1j * w * cast(el.value)) if not dc else zero
            elif el.kind == "I":
                f, t = ix(el.nodes[0]), ix(el.nodes[1])
                val = cast(el.value)
                if f is not None:
                    rhs[f] -= val
                if t is not None:
                    rhs[t] += val
            else:  # OA
                p, m, o = (ix(n) for n in el.nodes)
                k = bidx[el.name]
                add(o, k, -one)                      # op-amp delivers current k into the output
                add(k, p, one); add(k, m, -one)      # v+ - v- = 0

        x = _solve_linear(A, rhs, mode == "exact")
        volt = {self.ground: zero}
        for n in nodes:
            volt[n] = x[idx[n]]
        cur = {}
        for el in self._elements:
            if el.kind == "R":
                cur[el.name] = (volt[el.nodes[0]] - volt[el.nodes[1]]) / cast(el.value)
            elif el.kind == "C":
                cur[el.name] = zero if dc else \
                    (volt[el.nodes[0]] - volt[el.nodes[1]]) * 1j * w * cast(el.value)
            elif el.kind == "I":
                cur[el.name] = cast(el.value)
            else:
                cur[el.name] = x[bidx[el.name]]
        return Solution(self, omega, volt, cur)


class Thevenin(namedtuple("Thevenin", "v_th z_th")):
    __slots__ = ()

    @property
    def r_th(self):
        return self.z_th

    @property
    def i_n(self):
        if self.z_th == 0:
            raise CircuitError("no Norton equivalent: Thevenin impedance is zero")
        return self.v_th / self.z_th


Norton = namedtuple("Norton", "i_n z_n")      # i_n flows a -> b through the short


def _unique(circuit, base):
    name = base
    while name in circuit._by_name:
        name += "_"
    return name


def _port(circuit, a, b):
    a = str(a)
    b = circuit.ground if b is None else str(b)
    nodes = circuit.nodes()
    for n in (a, b):
        if n not in nodes:
            raise CircuitError("unknown node %r" % n)
    if a == b:
        raise CircuitError("the two port nodes are the same")
    return a, b


def _with_reference(circuit, b):
    # A bare two-terminal network may have no "0" node; use b as ground then.
    if circuit.ground in circuit.nodes():
        return circuit
    return circuit.copy(ground=b)


def _omega(freq):
    return 0 if not freq else 2.0 * math.pi * float(freq)


def thevenin(circuit, a, b=None, freq=0):
    """Thevenin equivalent at a (+) and b (default ground).

    z_th is found by zeroing the sources and pushing 1 A in at a and out at b.
    """
    a, b = _port(circuit, a, b)
    circuit = _with_reference(circuit, b)
    w = _omega(freq)
    v_oc = circuit.solve(w).v(a, b)
    dead = circuit.deactivated()
    dead.current_source(_unique(dead, "_test"), b, a, 1)
    z = dead.solve(w).v(a, b)
    return Thevenin(v_oc, z)


def norton(circuit, a, b=None, freq=0):
    # short the port with a 0 V source and read its current
    a, b = _port(circuit, a, b)
    circuit = _with_reference(circuit, b)
    w = _omega(freq)
    short = circuit.copy()
    name = _unique(short, "_short")
    short.voltage_source(name, a, b, 0)
    i_sc = short.solve(w).i(name)
    return Norton(i_sc, thevenin(circuit, a, b, freq).z_th)


_MEASURE = re.compile(
    r"^\s*(-?)\s*([vip])\(\s*([^,()\s]+)\s*(?:,\s*([^,()\s]+)\s*)?\)\s*(?::(abs|deg|re|im))?\s*$")


def measure(solution, expr):
    """Evaluate e.g. "v(out)", "v(a,b)", "-i(R1)" or "p(R2)" on a solution.
    Complex (AC) results need a suffix: :abs, :deg, :re or :im.
    """
    m = _MEASURE.match(expr) if isinstance(expr, str) else None
    if not m:
        raise CircuitError("cannot parse measure %r (examples: v(out), v(a,b):abs, -i(R1), p(R2))" % (expr,))
    neg, what, x, y, part = m.groups()
    if what == "v":
        val = solution.v(x, y)
    elif what == "i":
        if y is not None:
            raise CircuitError("i(...) takes one element name")
        val = solution.i(x)
    else:
        if y is not None:
            raise CircuitError("p(...) takes one element name")
        val = solution.power(x)
    if isinstance(val, complex) and part is None:
        raise CircuitError("result is complex: add :abs, :deg, :re or :im to %r" % (expr,))
    if part == "abs":
        val = abs(val)
    elif part == "deg":
        val = math.degrees(cmath.phase(val))
    elif part == "re":
        val = val.real
    elif part == "im":
        val = val.imag
    val = float(val)
    return -val if neg else val


def first_order_parameters(circuit, element):
    """(x_final, tau) for the one L or C in a circuit with DC sources.

    x_final is the final capacitor voltage or inductor current, tau is
    R_th*C or L/R_th with R_th taken across the storage element.
    """
    el = circuit.element(element)
    if el.kind not in ("L", "C"):
        raise CircuitError("%r is not an inductor or capacitor" % (element,))
    if sum(1 for e in circuit.elements if e.kind in ("L", "C")) != 1:
        raise CircuitError("circuit must contain exactly one L or C")
    a, b = el.nodes
    final = circuit.solve_dc()
    rest = circuit.without(element)
    r_th = thevenin(rest, a, b).z_th
    if r_th == 0:
        raise CircuitError("zero resistance seen by %s: time constant is zero or infinite" % element)
    if el.kind == "C":
        return final.v(a, b), r_th * el.value
    return final.i(element), el.value / r_th


def first_order_response(circuit, element, x0, t):
    x_final, tau = first_order_parameters(circuit, element)
    return first_order_step(x0, float(x_final), float(tau), t)


def series(*values):
    if not values:
        raise ValueError("series() needs at least one value")
    return sum(values)


def parallel(*values):
    # works for complex impedances too
    if not values:
        raise ValueError("parallel() needs at least one value")
    if any(v == 0 for v in values):
        return 0.0
    total = sum(1.0 / v for v in values)
    if total == 0:
        raise ValueError("parallel combination is an open circuit (infinite impedance)")
    return 1.0 / total


def polar(magnitude, angle_deg):
    return cmath.rect(magnitude, math.radians(angle_deg))


def to_polar(z):
    return abs(z), math.degrees(cmath.phase(z))


def voltage_divider(v_in, r_top, r_bottom):
    # unloaded; returns the voltage across r_bottom
    _positive(r_top, "r_top"); _positive(r_bottom, "r_bottom")
    return v_in * r_bottom / (r_top + r_bottom)


def current_divider(i_total, r_branch, *r_others):
    # i_total splits between r_branch and the parallel group r_others
    _positive(r_branch, "r_branch")
    if not r_others:
        return float(i_total)
    for r in r_others:
        _positive(r, "r_other")
    r_rest = parallel(*r_others)
    return i_total * r_rest / (r_branch + r_rest)


def power_in_load(v_th, r_th, r_load):
    _positive(r_th, "r_th"); _positive(r_load, "r_load")
    return v_th ** 2 * r_load / (r_th + r_load) ** 2


def max_power_transfer(v_th, r_th):
    # returns (r_load, p_max)
    _positive(r_th, "r_th")
    return r_th, v_th ** 2 / (4.0 * r_th)


def max_power_transfer_ac(v_th_rms, z_th, load="complex"):
    """Returns (z_load, p_max) for an RMS Thevenin source.

    load="complex": conjugate match, p = V^2 / (4 R_th).
    load="resistive": R_L = |z_th|. That delivers less than the conjugate
    match unless X_th is zero.
    """
    z = complex(z_th)
    if not z.real > 0:
        raise ValueError("Re(z_th) must be positive")
    v2 = _finite(abs(v_th_rms), "v_th_rms") ** 2
    if load == "complex":
        return z.conjugate(), v2 / (4.0 * z.real)
    if load == "resistive":
        rl = abs(z)
        return rl, v2 * rl / ((z.real + rl) ** 2 + z.imag ** 2)
    raise ValueError("load must be 'complex' or 'resistive'")


def time_constant_rc(r, c):
    return _positive(r, "r") * _positive(c, "c")


def time_constant_rl(l, r):
    return _positive(l, "l") / _positive(r, "r")


def first_order_step(x0, x_final, tau, t):
    _positive(tau, "tau")
    if _finite(t, "t") < 0:
        raise ValueError("t must be >= 0")
    return x_final + (x0 - x_final) * math.exp(-t / tau)


def first_order_time(x0, x_final, tau, x_target):
    _positive(tau, "tau")
    if x0 == x_final:
        raise ValueError("x0 equals x_final: nothing changes")
    frac = (x_target - x_final) / (x0 - x_final)
    if not 0 < frac <= 1:
        raise ValueError("target %r is not reached: it must lie between x0 and x_final" % (x_target,))
    return -tau * math.log(frac)


def time_to_fraction(tau, fraction):
    # time to cover `fraction` of the total change
    _positive(tau, "tau")
    if not 0 <= _finite(fraction, "fraction") < 1:
        raise ValueError("fraction must be in [0, 1)")
    return -tau * math.log(1.0 - fraction)


def tau_from_points(x0, x_final, x_t, t):
    # tau from a single reading x_t taken t seconds after the step
    _positive(t, "t")
    frac = (x_t - x_final) / (x0 - x_final)
    if not 0 < frac < 1:
        raise ValueError("x_t must lie strictly between x0 and x_final")
    return -t / math.log(frac)


def capacitor_energy(c, v):
    return 0.5 * _positive(c, "c") * v ** 2


def inductor_energy(l, i):
    return 0.5 * _positive(l, "l") * i ** 2


def impedance_inductor(l, f):
    return 1j * 2 * math.pi * _positive(f, "f") * _positive(l, "l")


def impedance_capacitor(c, f):
    return 1.0 / (1j * 2 * math.pi * _positive(f, "f") * _positive(c, "c"))


def series_rlc_impedance(r, l, c, f):
    return _positive(r, "r") + impedance_inductor(l, f) + impedance_capacitor(c, f)


def rms_sine(peak):
    return abs(_finite(peak, "peak")) / math.sqrt(2.0)


def rms_composite(dc, peaks):
    # Only valid when the sinusoids all have different frequencies; then the
    # cross terms average to zero and the squares just add.
    return math.sqrt(_finite(dc, "dc") ** 2 + sum(_finite(p, "peak") ** 2 / 2.0 for p in peaks))


def rc_corner_frequency(r, c):
    # -3 dB and -45 degrees
    return 1.0 / (2 * math.pi * time_constant_rc(r, c))


Resonance = namedtuple("Resonance", "f0 q bandwidth")   # bandwidth = f0/Q, half-power


def series_rlc_resonance(r, l, c):
    r, l, c = _positive(r, "r"), _positive(l, "l"), _positive(c, "c")
    f0 = 1.0 / (2 * math.pi * math.sqrt(l * c))
    q = math.sqrt(l / c) / r
    return Resonance(f0, q, f0 / q)


def parallel_rlc_resonance(r, l, c):
    # R, L and C all across the same two nodes, so Q = R*sqrt(C/L), the
    # reciprocal of the series form
    r, l, c = _positive(r, "r"), _positive(l, "l"), _positive(c, "c")
    f0 = 1.0 / (2 * math.pi * math.sqrt(l * c))
    q = r * math.sqrt(c / l)
    return Resonance(f0, q, f0 / q)


def resonant_capacitance(f0, l):
    return 1.0 / ((2 * math.pi * _positive(f0, "f0")) ** 2 * _positive(l, "l"))


def reactive_power(p, pf):
    # lagging load, so Q comes out positive
    return _finite(p, "p") * math.tan(math.acos(_pf(pf)))


def apparent_power(p, pf):
    return _finite(p, "p") / _pf(pf)


def line_current(p, v, pf, phases=1):
    # for phases=3, v is the line-to-line voltage
    p = _finite(p, "p"); v = _positive(v, "v"); pf = _pf(pf)
    if phases == 1:
        return p / (v * pf)
    if phases == 3:
        return p / (SQRT3 * v * pf)
    raise ValueError("phases must be 1 or 3")


def pf_correction_reactive(p, pf_old, pf_new):
    # var the capacitors must supply; both power factors lagging
    pf_old, pf_new = _pf(pf_old, "pf_old"), _pf(pf_new, "pf_new")
    if pf_new < pf_old:
        raise ValueError("pf_new must not be lower than pf_old")
    return _finite(p, "p") * (math.tan(math.acos(pf_old)) - math.tan(math.acos(pf_new)))


def pf_correction_capacitance(p, v, f, pf_old, pf_new, phases=1, connection="delta"):
    """Correction capacitance in farads; p is the total real power.

    For phases=3, v is line-to-line and the result is the value of each of
    the three capacitors. In delta each one sees V_L and supplies Qc/3; in wye
    each sees V_L/sqrt(3), so it needs three times the capacitance.
    """
    qc = pf_correction_reactive(p, pf_old, pf_new)
    w = 2 * math.pi * _positive(f, "f")
    v = _positive(v, "v")
    if phases == 1:
        return qc / (w * v * v)
    if phases == 3:
        if connection == "delta":
            return qc / (3 * w * v * v)
        if connection == "wye":
            return qc / (w * v * v)
        raise ValueError("connection must be 'delta' or 'wye'")
    raise ValueError("phases must be 1 or 3")


# RMS magnitudes; p, q and s are totals for all three phases
ThreePhase = namedtuple("ThreePhase", "v_phase i_phase i_line p q s")


def balanced_load(connection, v_line, z_phase):
    # z_phase is the (complex) impedance of one phase, v_line is line-to-line
    z = complex(z_phase)
    if z == 0 or z.real < 0:
        raise ValueError("z_phase must be non-zero with non-negative resistance")
    v_line = _positive(v_line, "v_line")
    if connection == "wye":
        vp = v_line / SQRT3
        ip = vp / abs(z)
        il = ip
    elif connection == "delta":
        vp = v_line
        ip = vp / abs(z)
        il = SQRT3 * ip
    else:
        raise ValueError("connection must be 'wye' or 'delta'")
    s = 3 * vp * ip
    phi = cmath.phase(z)
    return ThreePhase(vp, ip, il, s * math.cos(phi), s * math.sin(phi), s)


def three_phase_power(v_line, i_line, pf):
    # returns (P, Q, S), Q positive (lagging)
    s = SQRT3 * _positive(v_line, "v_line") * _positive(i_line, "i_line")
    pf = _pf(pf)
    return s * pf, s * math.sqrt(1 - pf * pf), s


def delta_to_wye(z_ab, z_bc, z_ca):
    # returns the wye arms (z_a, z_b, z_c)
    t = z_ab + z_bc + z_ca
    return z_ab * z_ca / t, z_ab * z_bc / t, z_bc * z_ca / t


def wye_to_delta(z_a, z_b, z_c):
    # returns (z_ab, z_bc, z_ca)
    s = z_a * z_b + z_b * z_c + z_c * z_a
    return s / z_c, s / z_a, s / z_b


def base_impedance(s_base, v_base):
    # three-phase: total VA and line-to-line voltage
    return _positive(v_base, "v_base") ** 2 / _positive(s_base, "s_base")


def base_current(s_base, v_base, phases=3):
    s, v = _positive(s_base, "s_base"), _positive(v_base, "v_base")
    if phases == 3:
        return s / (SQRT3 * v)
    if phases == 1:
        return s / v
    raise ValueError("phases must be 1 or 3")


def to_per_unit(value, base):
    return value / _positive(base, "base")


def from_per_unit(value_pu, base):
    return value_pu * _positive(base, "base")


def change_base(z_pu, s_old, v_old, s_new, v_new):
    return z_pu * (_positive(s_new, "s_new") / _positive(s_old, "s_old")) * \
        (_positive(v_old, "v_old") / _positive(v_new, "v_new")) ** 2


def saturate(v, v_pos, v_neg=None):
    v_neg = -v_pos if v_neg is None else v_neg
    if not v_neg < v_pos:
        raise ValueError("need v_neg < v_pos")
    return max(v_neg, min(v_pos, v))


def _limit(v, v_sat):
    return v if v_sat is None else saturate(v, _positive(v_sat, "v_sat"))


# Ideal op-amp stages. v_sat, if given, clips the output symmetrically.

def inverting_amp(v_in, r_in, r_f, v_sat=None):
    return _limit(-_positive(r_f, "r_f") / _positive(r_in, "r_in") * v_in, v_sat)


def non_inverting_amp(v_in, r_g, r_f, v_sat=None):
    # r_g goes from the inverting input to ground
    return _limit((1 + _positive(r_f, "r_f") / _positive(r_g, "r_g")) * v_in, v_sat)


def summing_amp(v_inputs, r_inputs, r_f, v_sat=None):
    if len(v_inputs) != len(r_inputs) or not v_inputs:
        raise ValueError("v_inputs and r_inputs must be non-empty and the same length")
    _positive(r_f, "r_f")
    total = sum(v / _positive(r, "r_input") for v, r in zip(v_inputs, r_inputs))
    return _limit(-r_f * total, v_sat)


def difference_amp(v1, v2, r1, r2, r3, r4, v_sat=None):
    """v1 -> r1 -> inverting input, r2 feedback; v2 -> r3 -> non-inverting
    input, r4 from there to ground. Resistors need not be matched; the usual
    (r2/r1)(v2 - v1) only holds when r4/r3 = r2/r1.
    """
    for name, r in (("r1", r1), ("r2", r2), ("r3", r3), ("r4", r4)):
        _positive(r, name)
    v_plus = v2 * r4 / (r3 + r4)
    return _limit(v_plus * (1 + r2 / r1) - v1 * r2 / r1, v_sat)


# Wrappers that problems.jsonl calls by name. Each one returns a float in SI
# units; verify.py does the conversion to the problem's unit.
METHODS = {}


def method(fn):
    METHODS[fn.__name__] = fn
    return fn


@method
def netlist_measure(netlist, measure_expr, freq=0):
    # freq=0 means DC
    c = Circuit.from_netlist(netlist, exact=True)
    sol = c.solve_ac(freq) if freq else c.solve_dc()
    return measure(sol, measure_expr)


@method
def thevenin_v(netlist, a, b="0"):
    c = Circuit.from_netlist(netlist, exact=True)
    return float(thevenin(c, a, b).v_th)


@method
def thevenin_r(netlist, a, b="0"):
    c = Circuit.from_netlist(netlist, exact=True)
    return float(thevenin(c, a, b).z_th)


@method
def norton_i(netlist, a, b="0"):
    c = Circuit.from_netlist(netlist, exact=True)
    return float(norton(c, a, b).i_n)


@method
def max_power_netlist(netlist, a, b="0"):
    c = Circuit.from_netlist(netlist, exact=True)
    th = thevenin(c, a, b)
    return max_power_transfer(float(th.v_th), float(th.z_th))[1]


@method
def load_power_netlist(netlist, a, b, r_load):
    # solved with the load in the circuit rather than through Thevenin
    c = Circuit.from_netlist(netlist, exact=True)
    name = _unique(c, "_load")
    c.resistor(name, a, b, Fraction(repr(r_load)))
    return float(c.solve_dc().power(name))


@method
def first_order_netlist(netlist, element, x0, t):
    c = Circuit.from_netlist(netlist, exact=True)
    return first_order_response(c, element, x0, t)


@method
def first_order_netlist_time(netlist, element, x0, x_target):
    c = Circuit.from_netlist(netlist, exact=True)
    x_final, tau = first_order_parameters(c, element)
    return first_order_time(x0, float(x_final), float(tau), x_target)


@method
def current_divider_branch(i_total, r_branch, r_others):
    return current_divider(i_total, r_branch, *r_others)


@method
def rc_voltage_at(v0, v_final, r, c, t):
    return first_order_step(v0, v_final, time_constant_rc(r, c), t)


@method
def rc_time_to_reach(v0, v_final, r, c, v_target):
    return first_order_time(v0, v_final, time_constant_rc(r, c), v_target)


@method
def rl_current_at(i0, i_final, r, l, t):
    return first_order_step(i0, i_final, time_constant_rl(l, r), t)


@method
def rl_time_to_reach(i0, i_final, r, l, i_target):
    return first_order_time(i0, i_final, time_constant_rl(l, r), i_target)


@method
def source_energy_rc_charge(c, v0, v_final):
    # The source moves charge C*(v_final - v0) at a constant v_final, whatever
    # the resistance. Charging from zero, half of it ends up in the resistor.
    return _positive(c, "c") * (v_final - v0) * v_final


@method
def series_rlc_impedance_abs(r, l, c, f):
    return abs(series_rlc_impedance(r, l, c, f))


@method
def series_rlc_impedance_phase_deg(r, l, c, f):
    return math.degrees(cmath.phase(series_rlc_impedance(r, l, c, f)))


@method
def resonance(kind, r, l, c, output):
    if kind == "series":
        res = series_rlc_resonance(r, l, c)
    elif kind == "parallel":
        res = parallel_rlc_resonance(r, l, c)
    else:
        raise ValueError("kind must be 'series' or 'parallel'")
    if output not in Resonance._fields:
        raise ValueError("output must be one of %s" % ", ".join(Resonance._fields))
    return getattr(res, output)


@method
def series_resonance_cap_voltage(v_source, r, l, c):
    # Found with an AC solve at f0 rather than as Q * V. The result is on the
    # same scale (RMS or peak) as v_source.
    f0 = series_rlc_resonance(r, l, c).f0
    ckt = Circuit()
    ckt.voltage_source("V1", "in", "0", v_source)
    ckt.resistor("R1", "in", "a", r)
    ckt.inductor("L1", "a", "b", l)
    ckt.capacitor("C1", "b", "0", c)
    return abs(ckt.solve_ac(f0).v("b"))


@method
def avg_power_composite(v_dc, v_peaks, r):
    return rms_composite(v_dc, v_peaks) ** 2 / _positive(r, "r")


@method
def avg_power_sine_peak(v_peak, r):
    return rms_sine(v_peak) ** 2 / _positive(r, "r")


@method
def load_power(v_th, r_th, r_load):
    return power_in_load(v_th, r_th, r_load)


@method
def max_power_ac(v_th_rms, r_th, x_th, load):
    return max_power_transfer_ac(v_th_rms, complex(r_th, x_th), load)[1]


@method
def tau_from_decay(x0, x_final, x_t, t):
    return tau_from_points(x0, x_final, x_t, t)


@method
def power_triangle_q(p, pf):
    return reactive_power(p, pf)


@method
def single_phase_line_current(p, v, pf):
    return line_current(p, v, pf, 1)


@method
def three_phase_line_current(p, v_line, pf):
    return line_current(p, v_line, pf, 3)


@method
def pf_capacitor(p, v, f, pf_old, pf_new, phases=1, connection="delta"):
    return pf_correction_capacitance(p, v, f, pf_old, pf_new, phases, connection)


@method
def pf_correction_var(p, pf_old, pf_new):
    return pf_correction_reactive(p, pf_old, pf_new)


@method
def current_after_correction(p, v, pf_old, pf_new):
    # despite the name, this is the drop in current, not the new current
    return line_current(p, v, pf_old) - line_current(p, v, pf_new)


@method
def balanced_load_output(connection, v_line, r, x, output):
    if output not in ThreePhase._fields:
        raise ValueError("output must be one of %s" % ", ".join(ThreePhase._fields))
    return getattr(balanced_load(connection, v_line, complex(r, x)), output)


@method
def wye_equivalent_line_current(v_line, r_delta, x_delta):
    z_y = complex(r_delta, x_delta) / 3.0
    return (_positive(v_line, "v_line") / SQRT3) / abs(z_y)


@method
def three_phase_load_voltage(v_line_source, r_line, x_line, r_load, x_load, load_connection):
    # Per-phase wye equivalent, solved at omega = 1 so that the reactances
    # can be entered directly as L = x and C = 1/|x| (see _series_z).
    z_load = complex(r_load, x_load)
    if load_connection == "delta":
        z_load = z_load / 3.0
    elif load_connection != "wye":
        raise ValueError("load_connection must be 'wye' or 'delta'")
    z_line = complex(r_line, x_line)
    ckt = Circuit()
    ckt.voltage_source("Van", "s", "0", _positive(v_line_source, "v_line_source") / SQRT3)
    _series_z(ckt, "ZL", "s", "m", z_line)
    _series_z(ckt, "ZD", "m", "0", z_load)
    return SQRT3 * abs(ckt.solve(1.0).v("m"))


def _series_z(ckt, name, a, b, z):
    # z = r + jx as R then L or C in series; only right for a solve at omega = 1
    z = complex(z)
    parts = []
    if z.real > 0:
        parts.append((ckt.resistor, "r", z.real))
    if z.imag > 0:
        parts.append((ckt.inductor, "x", z.imag))
    elif z.imag < 0:
        parts.append((ckt.capacitor, "x", -1.0 / z.imag))
    if not parts:
        ckt.voltage_source(name + "_short", a, b, 0)
        return
    pts = [a] + ["%s_n%d" % (name, k) for k in range(1, len(parts))] + [b]
    for k, (add, tag, val) in enumerate(parts):
        add("%s_%s" % (name, tag), pts[k], pts[k + 1], val)


@method
def ohm_to_pu(z_ohm, s_base, v_base):
    return to_per_unit(z_ohm, base_impedance(s_base, v_base))


@method
def fault_current_series(s_base, v_base, z_pu_list, v_pu=1.0):
    # all impedances on the same base, in series between source and fault
    z = sum(_positive(x, "z_pu") for x in z_pu_list)
    return v_pu / z * base_current(s_base, v_base, 3)


def get_method(name):
    # METHODS first, then lower-case public functions; KeyError otherwise
    if name in METHODS:
        return METHODS[name]
    if name in __all__ and callable(globals().get(name)) and name[:1].islower():
        return globals()[name]
    raise KeyError(name)
