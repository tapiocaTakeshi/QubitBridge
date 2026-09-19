"""The APQB state algebra against the identities the paper states."""

import math
import unittest

from qubitbridge import apqb


class TestInvariants(unittest.TestCase):
    RS = [-1.0, -0.97, -0.5, -1e-9, 0.0, 1e-9, 0.25, 0.5, 0.9, 1.0]

    def test_constraint_holds_for_every_constructor(self):
        for r in self.RS:
            self.assertAlmostEqual(
                apqb.APQBState.from_r(r).constraint_error(), 0.0, places=12)
        for a in (-40.0, -3.0, -0.1, 0.0, 0.1, 3.0, 40.0):
            self.assertAlmostEqual(
                apqb.APQBState.from_latent(a).constraint_error(), 0.0, places=12)
        for theta in (-1.5, -0.3, 0.0, 0.3, math.pi / 4, math.pi / 2):
            self.assertAlmostEqual(
                apqb.APQBState.from_theta(theta).constraint_error(), 0.0, places=12)

    def test_r_and_temperature_trade_off(self):
        """r^2 + T^2 = 1, the paper's confidence/randomness trade-off."""
        for r in self.RS:
            state = apqb.APQBState.from_r(r)
            self.assertAlmostEqual(state.r ** 2 + state.T ** 2, 1.0, places=12)

    def test_r_equals_cos_two_theta(self):
        for r in self.RS:
            state = apqb.APQBState.from_r(r)
            self.assertAlmostEqual(math.cos(2 * state.theta), state.r, places=12)
            self.assertAlmostEqual(math.sin(2 * state.theta), state.eta, places=12)

    def test_sech_is_stable_at_extremes(self):
        self.assertAlmostEqual(apqb.sech(0.0), 1.0, places=15)
        self.assertEqual(apqb.sech(1000.0), 0.0)
        self.assertEqual(apqb.sech(-1000.0), 0.0)

    def test_latent_encoding_matches_paper_eq12(self):
        for a in (-5.0, -0.7, 0.0, 0.7, 5.0):
            state = apqb.encode(a, "latent")
            self.assertAlmostEqual(state.r, math.tanh(a), places=14)
            self.assertAlmostEqual(state.eta, apqb.sech(a), places=14)


class TestEncodings(unittest.TestCase):
    def test_round_trip(self):
        cases = {
            "latent": [-3.0, -0.5, 0.0, 0.5, 3.0],
            "linear": [-1.0, -0.3, 0.0, 0.3, 1.0],
            "angle": [0.0, 0.3, math.pi / 4, math.pi / 2 - 0.01],
            "prob": [0.0, 0.25, 0.5, 0.75, 1.0],
        }
        for mode, values in cases.items():
            for value in values:
                got = apqb.decode(apqb.encode(value, mode), mode)
                self.assertAlmostEqual(got, value, places=8, msg=f"{mode} {value}")

    def test_prob_encoding_matches_measurement_probabilities(self):
        for p in (0.0, 0.3, 0.5, 1.0):
            p0, p1 = apqb.probabilities(apqb.encode(p, "prob"))
            self.assertAlmostEqual(p0, p, places=12)
            self.assertAlmostEqual(p0 + p1, 1.0, places=12)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            apqb.encode(0.0, "bogus")
        with self.assertRaises(ValueError):
            apqb.decode(apqb.ZERO_STATE, "bogus")


class TestOperations(unittest.TestCase):
    def test_interact_is_complex_multiplication(self):
        a, b = apqb.encode(0.7), apqb.encode(-1.3)
        got = apqb.interact(a, b)
        want = a.z * b.z
        self.assertAlmostEqual(got.r, want.real, places=14)
        self.assertAlmostEqual(got.eta, want.imag, places=14)

    def test_interact_adds_angles(self):
        a, b = apqb.encode(0.2, "angle"), apqb.encode(0.5, "angle")
        self.assertAlmostEqual(apqb.interact(a, b).theta, 0.7, places=12)

    def test_mul_multiplies_correlations_and_stays_canonical(self):
        a, b = apqb.encode(0.8, "linear"), apqb.encode(-0.5, "linear")
        out = apqb.state_mul(a, b)
        self.assertAlmostEqual(out.r, -0.4, places=14)
        self.assertGreaterEqual(out.eta, 0.0)
        self.assertAlmostEqual(out.constraint_error(), 0.0, places=12)

    def test_correlate_is_cosine_of_the_angle_difference(self):
        a, b = apqb.encode(0.3, "angle"), apqb.encode(0.9, "angle")
        self.assertAlmostEqual(apqb.correlate(a, b), math.cos(2 * (0.3 - 0.9)),
                               places=12)
        self.assertAlmostEqual(apqb.correlate(a, a), 1.0, places=12)

    def test_uncertainty_is_the_paper_temperature(self):
        for theta in (0.0, 0.2, math.pi / 4, math.pi / 2):
            state = apqb.encode(theta, "angle")
            self.assertAlmostEqual(apqb.uncertainty(state),
                                   abs(math.sin(2 * theta)), places=12)
        # Maximum randomness at theta = pi/4 (r = 0), none at the endpoints.
        self.assertAlmostEqual(apqb.uncertainty(apqb.encode(0.0, "linear")), 1.0,
                               places=12)
        self.assertAlmostEqual(apqb.uncertainty(apqb.encode(1.0, "linear")), 0.0,
                               places=12)

    def test_entropy_endpoints(self):
        self.assertAlmostEqual(apqb.entropy_z(apqb.encode(0.0, "linear")), 1.0,
                               places=12)
        self.assertAlmostEqual(apqb.entropy_z(apqb.encode(1.0, "linear")), 0.0,
                               places=12)
        self.assertAlmostEqual(apqb.entropy_z(apqb.encode(-1.0, "linear")), 0.0,
                               places=12)

    def test_gate_is_identity_at_zero_coupling(self):
        target, source = apqb.encode(0.4), apqb.encode(-0.8)
        self.assertEqual(apqb.gate(target, source, 0.0), target)

    def test_gate_stays_in_range_for_large_coupling(self):
        target, source = apqb.encode(0.99, "linear"), apqb.encode(1.0, "linear")
        out = apqb.gate(target, source, 50.0)
        self.assertLessEqual(abs(out.r), 1.0)
        self.assertAlmostEqual(out.constraint_error(), 0.0, places=12)

    def test_power_matches_z_to_the_k(self):
        state = apqb.encode(0.35, "linear")
        for k in range(0, 6):
            got = apqb.power(state, k)
            want = state.z ** k
            self.assertAlmostEqual(got.r, want.real, places=12)
            self.assertAlmostEqual(got.eta, want.imag, places=12)

    def test_power_gives_the_chebyshev_features_of_prop2(self):
        """Re(z^k) = T_k(r) and Im(z^k) = eta * U_{k-1}(r)."""
        for r in (-0.9, -0.2, 0.0, 0.4, 0.95):
            state = apqb.APQBState.from_r(r)
            phi = math.acos(r)
            for k in range(1, 6):
                zk = apqb.power(state, k)
                self.assertAlmostEqual(zk.r, math.cos(k * phi), places=10)
                u = math.sin(k * phi) / math.sin(phi) if abs(r) < 1 else k
                self.assertAlmostEqual(zk.eta, state.eta * u, places=10)

    def test_power_rejects_negative_degree(self):
        with self.assertRaises(ValueError):
            apqb.power(apqb.ZERO_STATE, -1)

    def test_measure_sample_follows_the_born_rule(self):
        state = apqb.encode(0.5, "linear")   # P(0) = 0.75
        draws = [apqb.measure_sample(state, u / 1000.0) for u in range(1000)]
        self.assertAlmostEqual(sum(1 for d in draws if d > 0) / 1000.0, 0.75,
                               places=2)

    def test_normalize_repairs_drift(self):
        drifted = apqb.APQBState(3.0, 4.0).normalized()
        self.assertAlmostEqual(drifted.r, 0.6, places=14)
        self.assertAlmostEqual(drifted.eta, 0.8, places=14)
        self.assertEqual(apqb.APQBState(0.0, 0.0).normalized(), apqb.ZERO_STATE)


if __name__ == "__main__":
    unittest.main()
