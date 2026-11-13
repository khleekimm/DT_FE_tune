"""Numerical checks on the claims the manuscript makes about the estimator."""
import unittest
from pathlib import Path
import sys

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from fetune.mechanical import (MechanicalModel, Objective, band_indices, fit,    # noqa: E402
                               completion, denominator_poles,
                               modal_resolvent, rank_report, reconstruction,
                               scales, score, selection_rule, spatial_candidates)


def system(seed=0, r=7, h=4, m=3, p=2, lines=25):
    rng = np.random.default_rng(seed)
    f = np.linspace(1., 20., lines)
    S = rng.normal(size=(m, h))
    B = rng.normal(size=(h, p))
    E = np.eye(h)
    Q = rng.normal(size=(h, h))
    Q = Q@Q.T
    W = rng.normal(size=(h, r))
    omega = np.sort(2*np.pi*rng.uniform(2, 18, r))
    chol = np.tril(rng.normal(size=(r, r))*.1)
    np.fill_diagonal(chol, abs(np.diag(chol))+.2)
    return f, S, B, E, Q, W, omega, chol@chol.T, rng


class ModalExpansion(unittest.TestCase):
    def test_matches_dense_resolvent(self):
        f, S, B, E, Q, W, omega, gamma, _ = system()
        r = len(omega)
        ref = 2*np.pi*max(f)
        lam, phi = modal_resolvent(omega/ref, gamma/ref)
        for s in 2j*np.pi*f/ref:
            dense = la.inv(s**2*np.eye(r)+s*gamma/ref+np.diag((omega/ref)**2))
            modal = (phi*(1/(s-lam))[None, :])@phi.T
            self.assertLess(la.norm(dense-modal)/la.norm(dense), 1e-10)

    def test_lancaster_identities(self):
        """Phi Phi^T = 0, Phi Lambda Phi^T = I and Phi Lambda^2 Phi^T = -Gamma."""
        f, S, B, E, Q, W, omega, gamma, _ = system(seed=3)
        ref = 2*np.pi*max(f)
        lam, phi = modal_resolvent(omega/ref, gamma/ref)
        self.assertLess(la.norm(phi@phi.T), 1e-9)
        self.assertLess(la.norm((phi*lam[None])@phi.T-np.eye(len(omega))), 1e-9)
        self.assertLess(la.norm(-(phi*lam[None]**2)@phi.T-gamma/ref), 1e-9)

    def test_residues_have_rank_one(self):
        """Proposition 2: every modal residue of the port transfer is rank one."""
        f, S, B, E, Q, W, omega, gamma, _ = system(seed=5)
        model = MechanicalModel(E@W, omega, gamma)
        ref = max(omega)
        lam, phi = modal_resolvent(omega/ref, gamma/ref)
        for j in range(len(lam)):
            residue = np.outer(S@model.Z@phi[:, j], (model.Z.T@B).T@phi[:, j])
            sv = la.svdvals(residue)
            self.assertLess(sv[1], 1e-10*sv[0])

    def test_feedthrough_is_port_inverse_mass(self):
        f, S, B, E, Q, W, omega, gamma, _ = system(seed=6)
        model = MechanicalModel(E@W, omega, gamma)
        high = model.evaluate([4e4], S, B)[0]
        direct = (S@model.Z)@(model.Z.T@B)
        self.assertLess(la.norm(high-direct)/la.norm(direct), 1e-3)


class Completeness(unittest.TestCase):
    def test_general_second_order_system_is_in_the_family(self):
        """Proposition 1: arbitrary M, C, K reduce to unit mass, diagonal Omega^2."""
        rng = np.random.default_rng(11)
        r = 6
        M = rng.normal(size=(r, r))
        M = M@M.T+r*np.eye(r)
        K = rng.normal(size=(r, r))
        K = K@K.T+r*np.eye(r)
        C = rng.normal(size=(r, r))
        C = C@C.T+r*np.eye(r)
        G, H = rng.normal(size=(3, r)), rng.normal(size=(r, 2))
        root = la.sqrtm(M).real
        inv = la.inv(root)
        evals, U = la.eigh(inv@K@inv)
        V = inv@U
        gamma, omega = U.T@inv@C@inv@U, np.sqrt(evals)
        for s in 2j*np.pi*np.linspace(1., 30., 9):
            a = s**2*G@la.solve(s**2*M+s*C+K, H)
            b = s**2*(G@V)@la.solve(s**2*np.eye(r)+s*gamma+np.diag(omega**2), V.T@H)
            self.assertLess(abs(a-b).max()/abs(a).max(), 1e-9)
        self.assertGreater(la.eigvalsh(gamma)[0], 0)


class Gradients(unittest.TestCase):
    def test_analytic_gradient_matches_finite_differences(self):
        f, S, B, E, Q, W, omega, gamma, rng = system(seed=1)
        y = (rng.normal(size=(len(f), S.shape[0], B.shape[1]))
             + 1j*rng.normal(size=(len(f), S.shape[0], B.shape[1])))
        for width in (0, 2, len(omega)-1):
            obj = Objective(f, y, S, B, E, Q, len(omega), width)
            x = obj.pack(W, omega, gamma)
            _, analytic = obj(x)
            numeric = np.zeros_like(analytic)
            for i in range(len(x)):
                step = np.zeros_like(x)
                step[i] = 1e-6
                numeric[i] = (obj(x+step)[0]-obj(x-step)[0])/2e-6
            self.assertLess(la.norm(analytic-numeric)/la.norm(numeric), 1e-5, width)

    def test_transfer_matches_direct_solve(self):
        f, S, B, E, Q, W, omega, gamma, rng = system(seed=2)
        y = np.zeros((len(f), S.shape[0], B.shape[1]), complex)
        obj = Objective(f, y, S, B, E, Q, len(omega), 3)
        x = obj.pack(W, omega, gamma)
        Wu, wu, gu, _ = obj.unpack(x)
        direct = np.array([s**2*(S@E@Wu)@la.solve(
            s**2*np.eye(len(omega))+s*gu+np.diag(wu**2), (B.T@E@Wu).T) for s in obj.s])
        self.assertLess(la.norm(obj.transfer(x)-direct)/la.norm(direct), 1e-10)

    def test_packing_round_trip_and_bandwidth(self):
        f, S, B, E, Q, W, omega, gamma, _ = system(seed=4)
        r = len(omega)
        obj = Objective(f, np.zeros((len(f), S.shape[0], B.shape[1]), complex),
                        S, B, E, Q, r, r-1)
        back = obj.unpack(obj.pack(W, omega, gamma))
        self.assertLess(la.norm(back[2]*obj.ref-gamma)/la.norm(gamma), 1e-12)
        narrow = Objective(f, np.zeros((len(f), S.shape[0], B.shape[1]), complex),
                           S, B, E, Q, r, 2)
        _, _, banded, _ = narrow.unpack(np.zeros(narrow.p*r+r+len(narrow.tri[0])))
        rows, cols = np.indices(banded.shape)
        self.assertLess(abs(banded[abs(rows-cols) > 2]).max(), 1e-12)


class Realisation(unittest.TestCase):
    def test_state_space_and_evaluation_agree(self):
        f, S, B, E, Q, W, omega, gamma, _ = system(seed=7)
        model = MechanicalModel(E@W, omega, gamma)
        checks = model.checks(np.linspace(1., 20., 9), S, B)
        self.assertLess(checks['state_space_relative'], 1e-9)
        self.assertLess(checks['evaluation_relative'], 1e-9)
        self.assertLess(checks['maximum_pole_real'], 0.)
        self.assertTrue(checks['positive_stiffness'])

    def test_force_response_is_linear_and_calibrated(self):
        f, S, B, E, Q, W, omega, gamma, _ = system(seed=8)
        model = MechanicalModel(E@W, omega, gamma)
        H = model.evaluate([7.], S, B)[0]
        load = np.array([3., -2j])
        self.assertLess(la.norm(H@(2*load)-2*(H@load)), 1e-12*la.norm(H@load))

    def test_same_map_carries_force_and_response(self):
        """Virtual work: the fitted Z appears on both sides, so swapping ports
        transposes the transfer of the square sub-problem."""
        f, S, B, E, Q, W, omega, gamma, _ = system(seed=9, m=3, p=3)
        model = MechanicalModel(E@W, omega, gamma)
        ports = np.asarray(S)
        direct = model.evaluate([5.], ports, ports.T)[0]
        self.assertLess(la.norm(direct-direct.T)/la.norm(direct), 1e-10)


class Spatial(unittest.TestCase):
    def setUp(self):
        n = 12
        inc = np.diff(np.eye(n), axis=0)
        self.n = n
        self.L = sp.csr_matrix(inc.T@inc)
        self.x = np.arange(n, dtype=float)*.1
        self.rigid = np.c_[np.ones(n), self.x]
        self.S = np.eye(n)[[0, 3, 6, 9, 11]]

    def test_completion_anchors_ports_and_reproduces_rigid(self):
        E, info = completion(self.L, sp.csr_matrix(self.S), self.rigid)
        self.assertLess(info['anchor_error'], 1e-8)
        self.assertLess(info['rigid_error'], 1e-8)
        self.assertEqual(info['port_rank'], self.S.shape[0])

    def test_port_rank_bounds_every_fitted_map(self):
        """Theorem: rank(Z) <= h for any W, independently of n."""
        E, _ = completion(self.L, sp.csr_matrix(self.S), self.rigid)
        rng = np.random.default_rng(0)
        for r in (3, 5, 9, 20):
            Z = E@rng.normal(size=(E.shape[1], r))
            self.assertLessEqual(rank_report(Z)['rank'], self.S.shape[0])

    def test_reconstruction_preserves_retained_rows(self):
        P, keep, drop, info = reconstruction(self.L, [0, 3, 6, 9, 11], self.rigid)
        self.assertLess(info['retained_identity_error'], 1e-10)
        self.assertLess(info['rigid_error'], 1e-8)
        self.assertLess(info['constant_error'], 1e-8)

    def test_unanchored_nullspace_is_rejected(self):
        with self.assertRaises(ValueError):
            completion(self.L, sp.csr_matrix(np.zeros((1, self.n))), None)

    def test_candidate_gates_report_rank_failure(self):
        E, _ = completion(self.L, sp.csr_matrix(self.S), self.rigid)
        rng = np.random.default_rng(1)
        r = 9
        model = MechanicalModel(E@rng.normal(size=(E.shape[1], r)),
                                np.sort(2*np.pi*rng.uniform(2, 18, r)),
                                np.diag(rng.uniform(.5, 2., r)))
        f = np.linspace(1., 20., 21)
        S = np.eye(self.n)[[0, 3, 6, 9, 11]]
        B = np.eye(self.n)[:, [2]]
        y = model.evaluate(f, S, B)
        rows, arrays, k = spatial_candidates(
            model, self.L, S, B, f, y, scales(y), [0, 2, 3, 6, 9, 11], [6, 8, 10],
            self.rigid)
        self.assertIsNone(k)
        self.assertTrue(all(not v['rank']['full_column_rank'] for v in rows))


class Estimation(unittest.TestCase):
    def test_recovers_a_known_classically_damped_system(self):
        rng = np.random.default_rng(12)
        n, r = 10, 4
        inc = np.diff(np.eye(n), axis=0)
        L = sp.csr_matrix(inc.T@inc)
        rigid = np.c_[np.ones(n), np.arange(n, dtype=float)]
        S = np.eye(n)[[0, 2, 4, 6, 8, 9]]
        B = np.eye(n)[:, [3, 7]]
        E, _ = completion(L, sp.vstack([sp.csr_matrix(S), sp.csr_matrix(B.T)]), rigid)
        truth = MechanicalModel(E@rng.normal(size=(E.shape[1], r)),
                                np.sort(2*np.pi*rng.uniform(3, 15, r)),
                                np.diag(rng.uniform(1., 3., r)))
        f = np.arange(1., 20., .05)
        y = truth.evaluate(f, S, B)
        model, info = fit(f, y, S, B, E, L.toarray(), r, 0, alpha=0., maxiter=4000)
        self.assertLess(info['training']['complex'], 1e-3)
        self.assertLess(
            abs(np.sort(model.omega)-np.sort(truth.omega)).max()/truth.omega.max(),
            1e-3)

    def test_denominator_relocation_finds_planted_poles(self):
        f = np.arange(1., 40., .02)
        s = 2j*np.pi*f
        poles = np.array([-.4+2j*np.pi*8, -.6+2j*np.pi*17, -.5+2j*np.pi*26])
        y = np.zeros((len(f), 2, 1), complex)
        rng = np.random.default_rng(2)
        for p in poles:
            res = rng.normal(size=(2, 1))
            y += res[None]/(s-p)[:, None, None]+res[None]/(s-p.conj())[:, None, None]
        found = denominator_poles(f, y, 3, iterations=8, decimate=1)
        self.assertLess(abs(np.sort(found.imag)-np.sort(poles.imag)).max()/(2*np.pi), .05)


class Selection(unittest.TestCase):
    def test_is_invariant_to_frequency_grouping(self):
        rng = np.random.default_rng(13)
        pred = rng.normal(size=(3, 40, 2, 2))
        sel = rng.normal(size=(4, 40, 2, 2))
        scale = np.ones((2, 2))
        labels = [dict(r=v, width=0) for v in (2, 4, 6)]
        a = selection_rule(pred, sel, scale, labels, np.repeat(np.arange(4), 10))
        b = selection_rule(pred, sel, scale, labels, np.repeat(np.arange(8), 5))
        self.assertEqual(a['best']['r'], b['best']['r'])
        for u, v in zip(a['candidates'], b['candidates']):
            self.assertAlmostEqual(u['selection'], v['selection'], places=12)

    def test_prefers_the_smallest_admissible_candidate(self):
        rng = np.random.default_rng(14)
        truth = rng.normal(size=(30, 2, 2))
        sel = truth[None]+.001*rng.normal(size=(4, 30, 2, 2))
        pred = np.stack([truth+.5*rng.normal(size=truth.shape), truth, truth])
        labels = [dict(r=v, width=0) for v in (2, 4, 6)]
        rule = selection_rule(pred, sel, np.ones((2, 2)), labels,
                              np.repeat(np.arange(5), 6))
        self.assertEqual(rule['selected']['r'], 4)


class Metrics(unittest.TestCase):
    def test_scales_have_a_floor_and_score_is_zero_on_equality(self):
        y = np.zeros((10, 2, 2), complex)
        y[:, 0, 0] = 1.
        self.assertGreater(scales(y).min(), 0.)
        self.assertEqual(score(y, y, scales(y))['complex'], 0.)

    def test_band_indices_respect_the_bandwidth(self):
        rows, cols = band_indices(6, 2)
        self.assertTrue(np.all(rows-cols <= 2))
        self.assertTrue(np.all(rows >= cols))
        self.assertEqual(len(band_indices(6, 5)[0]), 21)


if __name__ == '__main__':
    unittest.main()
