# circuit-problems

A set of 54 circuit-analysis problems (DC, AC, transients, resonance, power factor, three-phase, per-unit, op-amps) and the small solver I use to check them. Every stored answer is recomputed from the code, so a wrong number or a typo in a parameter shows up as a failure instead of sitting quietly in the file.

It only uses the Python standard library. Python 3.9 or newer.

## Running it

```
python3 verify.py                        # check every problem
python3 verify.py --topic opamp          # one topic
python3 verify.py --list                 # ids, topics, difficulty
python3 verify.py --show res_mesh_direction
python3 verify.py --check res_mesh_direction 0.5
python3 -m unittest -v                   # tests
```

`--show` prints the statement and then the worked solution. `--check ID VALUE` takes an answer in the problem's own unit and says PASS or FAIL; if the value matches one of the listed wrong answers it also tells you which mistake it is. Exit codes: 0 all good, 1 a problem failed or the file is broken, 2 bad command line (unknown id, unreadable file, value that isn't a number).

## What's in here

`problems.jsonl` has one problem per line. Each one has a statement, the answer with its unit and tolerance, a worked solution, one to three typical wrong answers with the reason they're wrong, and a few grading criteria. The numbers and scenarios are made up for this repo. A lot of the problems are built around one specific trap: RMS vs peak, line vs phase, delta vs wye, the sign of a current, or a mixed-up mA/kΩ/µF prefix.

`circuits.py` is the solver. It does modified nodal analysis on a netlist, at DC or at a single AC frequency, plus the usual closed-form formulas (dividers, Thevenin/Norton, max power transfer, RC/RL steps, resonance, pf correction, three-phase, per-unit, op-amp stages).

`verify.py` loads the problem file, checks its structure, reruns each problem's method with its parameters and compares the result with the stored answer.

What's covered: ideal sources, R, L, C and ideal op-amps, sinusoidal steady state at one frequency, first-order transients. No diodes, transistors, transformers or time-domain simulation.

## Using the solver directly

```python
from circuits import Circuit, thevenin, measure, polar, series_rlc_resonance

c = Circuit()
c.voltage_source("V1", "in", "0", 24)        # v(in) - v(0) = 24 V
c.resistor("R1", "in", "out", 4700)
c.resistor("R2", "out", "0", 10000)
s = c.solve_dc()
s.v("out")                                   # Fraction(800, 49), exact since all values are ints
th = thevenin(c, "out")                      # th.v_th, th.z_th

ac = Circuit()
ac.voltage_source("V1", "in", "0", polar(230, 0))     # 230 V RMS at 0 degrees
ac.resistor("R1", "in", "a", 40)
ac.inductor("L1", "a", "0", 0.15)
sol = ac.solve_ac(50)                        # 50 Hz
measure(sol, "i(R1):abs"), measure(sol, "i(R1):deg")  # 3.721 A, -49.67 degrees

series_rlc_resonance(8, 2e-3, 0.5e-6)        # Resonance(f0=5032.9, q=7.906, bandwidth=636.6)
```

Things to know about the solver:

* SI units throughout. Angles are radians unless the name ends in `_deg`.
* An element's current is measured from its first node to its second, through the element. So a voltage source that is delivering power has a negative current, and `power(name)` (power absorbed) is negative for it too.
* `current_source(name, from_node, to_node, amps)` pushes the current from `from_node` to `to_node` through the source.
* With only `int` or `Fraction` values a DC solve is exact (`Fraction`); otherwise it's float. AC results are `complex`. If you give RMS phasors you get RMS results and power in W, var and VA.
* `solve_dc` treats inductors as shorts and capacitors as opens.
* The op-amp in the solver never saturates. The closed-form stages take a `v_sat` argument if you need clipping.
* A circuit with no unique solution (floating node, parallel voltage sources, a cut-set of current sources, an op-amp without feedback) raises `SingularCircuitError` and the message says why.

## Problem format

One JSON object per line:

| field | meaning |
| --- | --- |
| `id` | unique identifier |
| `topic` | one of the topics below |
| `difficulty` | 1 (one formula), 2 (a few steps), 3 (several steps with a catch) |
| `statement` | self-contained, SI units, says what to report and in which unit |
| `method` | function in `circuits.py` that computes the answer (from `METHODS` or a public function) |
| `params` | keyword arguments for that function |
| `answer` | reference answer in `unit`, rounded to 4 significant figures |
| `unit` | e.g. `V`, `mA`, `uF`, `kvar`, `ohm`, `deg`, `pu`, `ratio` |
| `rel_tol` | relative tolerance for accepting an answer |
| `solution` | numbered steps in one string |
| `common_mistakes` | 1 to 3 items `{mistake, wrong_answer, why}`; `wrong_answer` is `null` when the mistake doesn't lead to one particular number |
| `criteria` | 3 to 5 items `{id, text, weight}` for marking a written solution; weights are positive |

Methods return SI values and `verify.py` converts to `unit` (a prefix `p n u m k M G` on a base unit). Circuit problems put a `netlist` in `params`, with rows like `["R", "R1", "a", "b", 1000]`, `["V", "V1", "in", "0", 12]`, `["I", "I1", "0", "a", 0.002]` or `["OA", "U1", "non_inv", "inv", "out"]`. A source value can be `[magnitude, angle_deg]` for an AC phasor. The quantity to report is picked with an expression such as `v(out)`, `v(a,b):abs`, `i(R1):deg` or `-i(V2)`.

## What the verifier checks

For each problem:

1. The schema: required fields present, no unknown ones, unique ids, difficulty 1 to 3, a unit it knows, `rel_tol` in (0, 0.5], a solution with at least two numbered steps, 1 to 3 mistakes, 3 to 5 criteria with distinct ids and positive weights.
2. It runs `method(**params)` and converts the result to `unit`.
3. `|computed - answer| <= rel_tol * |answer| (+ abs_tol)`.
4. No `wrong_answer` falls inside the tolerance of the right answer, so a listed mistake can't be graded as correct.

Bad lines, unknown methods, bad parameters and singular circuits come out as a FAIL with a message, not a traceback.

A caveat: this proves the stored answers can be reproduced from the code and that the file is well formed. It does not check that the statement text or the worked solution actually match the answer. When the set was written, the answers were also re-derived without `circuits.py` (relaxation on the resistor networks, numerical integration of the transients, frequency sweeps, plain phasor arithmetic). Those were throwaway scripts and aren't in the repo. Tolerances are 1%, which leaves room for rounding in someone's working but still separates the listed mistakes from a correct answer.

## Topics

| topic | problems | contents |
| --- | --- | --- |
| `resistive` | 10 | loaded divider, current divider, bridge with detector, quarter bridge, superposition (voltage and power), node analysis with a current source, branch current direction, ladder equivalent resistance, delta-wye bridge |
| `thevenin` | 6 | Thevenin voltage and resistance, Norton current, maximum power (DC, AC with complex load, AC with resistive load) |
| `transients` | 6 | RC and RL step response, time to a threshold, RC and RL circuits reduced with the solver, energy drawn from the source while charging |
| `ac` | 7 | series RLC impedance, current phase, RC low-pass magnitude and corner frequency, series-parallel phasor network, DC plus sine power, parallel RL impedance |
| `resonance` | 5 | series f0 and Q, parallel bandwidth, capacitor voltage at resonance, tuning capacitor |
| `power_factor` | 4 | reactive power, correcting capacitor (single phase and three-phase delta), line current reduction |
| `three_phase` | 6 | wye and delta line current, real power, delta-to-wye equivalent, current from power and power factor, feeder voltage drop |
| `per_unit` | 4 | base current, ohms to per unit, change of base, fault current |
| `opamp` | 6 | non-inverting, summing, unmatched difference amplifier, saturation, source resistance, two-stage cascade |

By difficulty: 10 at level 1, 32 at level 2, 12 at level 3.

## Tests

`test_circuits.py` compares the solver with results worked out by hand or another way: a Wheatstone bridge (detector current exactly 1/3025 A), superposition, a five-resistor bridge via delta-wye, series RLC impedance, resonance band edges, Thevenin/Norton, an RC step against a small time-step simulation, and a few hundred random circuits checked for KCL and the element laws. It also feeds it floating nodes, source loops and assorted bad input.

`test_verify.py` checks that the shipped file passes, that corrupting an answer or a parameter makes it fail, that each kind of schema error is caught without a traceback, and that the command-line options and exit codes behave.

## Author

Ali Doğaner. MIT licensed.
