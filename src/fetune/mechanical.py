"""Reciprocal second-order identification from calibrated accelerance.

One displacement map enters both force participation, through virtual work,
and response sampling:

    u = T q,   T = E W,   q'' + Gamma q' + Omega^2 q = T^T B f,   a = S T q''.

Same-order mass normalization and stiffness diagonalization are unrestricted
coordinate changes. Fixed E, finite order, damping sparsity and bounds are
additional estimator restrictions. The legacy storage key Z denotes T during
fitting, or retained rows Z_R when an optional reconstruction P is supplied.

The resolvent is expanded in the symmetric quadratic eigenvalue problem rather
than inverted line by line, and every second-order gradient term is assembled
from first-order divided differences.  Both costs are therefore linear in the
number of frequency lines, which is what makes a modally dense band affordable.

No generating mass, damping or stiffness matrix, exact mode, withheld response
or previously fitted model is an argument to fit().  All exported transfers are
accelerance in SI units.  The geometry prior is explicitly rank limited; no
invisible columns are added.
"""
from dataclasses import dataclass
import time
import warnings
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.optimize import least_squares, minimize
from scipy.signal import find_peaks, peak_widths
from scipy.signal._peak_finding_utils import PeakPropertyWarning


# --------------------------------------------------------------------------
# reporting metrics
# --------------------------------------------------------------------------
def scales(y):
    """Training channel RMS with a relative and an absolute floor."""
    rms = np.sqrt(np.mean(abs(y)**2, axis=0))
    return np.maximum(rms, max(1e-12, .001*np.median(rms)))


def score(h, y, scale):
    e = np.mean(abs((h-y)/scale)**2, axis=0)
    phase = abs(np.angle(h*y.conj(), deg=True))
    return dict(complex=float(np.sqrt(e.mean())), per_channel=np.sqrt(e).tolist(),
                worst=float(np.sqrt(e.max())),
                absolute_rms=np.sqrt(np.mean(abs(h-y)**2, axis=0)).tolist(),
                magnitude=float(np.sqrt(np.mean(((abs(h)-abs(y))/scale)**2))),
                phase_median_deg=np.median(phase, axis=0).tolist(),
                phase_p90_deg=np.quantile(phase, .9, axis=0).tolist())


def rank_report(z, mass_weights=None, rtol=1e-7):
    """Geometry quadrature and unit column scaling, not fitted target gains."""
    a = np.asarray(z, float)
    if mass_weights is not None:
        a = np.sqrt(mass_weights)[:, None]*a
    col = la.norm(a, axis=0)
    a = a/np.maximum(col, 1e-30)
    sv = la.svdvals(a)
    rank = int(np.sum(sv > sv[0]*rtol)) if len(sv) and sv[0] else 0
    return dict(rank=rank, columns=z.shape[1], scaled_singular_values=sv.tolist(),
                column_norms=col.tolist(), threshold=rtol,
                condition=float(sv[0]/sv[-1]) if rank == z.shape[1] else None,
                full_column_rank=rank == z.shape[1])


# --------------------------------------------------------------------------
# spatial completion and reconstruction
# --------------------------------------------------------------------------
def completion(Q, A, rigid=None):
    """Constrained common completion from independent physical port rows.

    Rigid reproduction is an explicit constraint on the map, never a
    mode-dependent hidden field.  An unanchored nullspace is an error, not
    something to be repaired by an invisible ridge.
    """
    A = sp.csr_matrix(A)
    Q = sp.csc_matrix(Q)
    gram = (A@A.T).toarray()
    _, rr, piv = la.qr(gram, pivoting=True)
    rank = int(np.sum(abs(np.diag(rr)) > max(abs(rr[0, 0])*1e-11, 1e-15)))
    ids = np.sort(piv[:rank])
    a = A[ids]
    kkt = sp.bmat([[Q, a.T], [a, None]], format='csc')
    rhs = np.vstack([np.zeros((Q.shape[0], rank)), np.eye(rank)])
    try:
        e = sla.splu(kkt).solve(rhs)[:Q.shape[0]]
    except RuntimeError as exc:
        raise ValueError('unanchored Q nullspace or dependent constraints') from exc
    before = None
    if rigid is not None:
        g = np.asarray(rigid)
        ag = a@g
        if np.linalg.matrix_rank(ag) < g.shape[1]:
            raise ValueError('ports do not determine the declared rigid representation')
        before = float(la.norm(e@ag-g)/la.norm(g))
        e += (g-e@ag)@la.pinv(ag)
    info = dict(independent_port_rows=ids.tolist(), port_rank=rank,
                anchor_error=float(la.norm(a@e-np.eye(rank))),
                rigid_error_before=before,
                rigid_error=float(la.norm(e@(a@rigid)-rigid)/la.norm(rigid))
                if rigid is not None else None,
                spatial_rank_bound=rank)
    return e, info


def reconstruction(Q, retained, rigid=None):
    """P = [I; -Q_dd^-1 Q_dc], plus declared kinematic reproduction."""
    Q = sp.csc_matrix(Q)
    n = Q.shape[0]
    keep = np.array(sorted(set(retained)), int)
    drop = np.setdiff1d(np.arange(n), keep)
    p = np.zeros((n, len(keep)))
    p[keep] = np.eye(len(keep))
    if len(drop):
        try:
            p[drop] = -sla.splu(Q[drop][:, drop]).solve(Q[drop][:, keep].toarray())
        except RuntimeError as exc:
            raise ValueError('retain anchors in every unanchored connected component') from exc
        if rigid is not None:
            if np.linalg.matrix_rank(rigid[keep]) < rigid.shape[1]:
                raise ValueError('retained coordinates do not span supported rigid motions')
            p[drop] += (rigid[drop]-p[drop]@rigid[keep])@la.pinv(rigid[keep])
    return p, keep, drop, dict(
        k=len(keep), dropped=len(drop),
        retained_identity_error=float(la.norm(p[keep]-np.eye(len(keep)))),
        constant_error=float(la.norm(p@np.ones(len(keep))-1)/np.sqrt(n)),
        rigid_error=float(la.norm(p@rigid[keep]-rigid)/la.norm(rigid))
        if rigid is not None else None)


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------
def modal_resolvent(omega, Gamma):
    """Symmetric expansion  (s^2 I + s Gamma + Omega^2)^-1 = Phi (s-L)^-1 Phi^T.

    Phi carries the Lancaster normalisation phi^T (2 lambda I + Gamma) phi = 1.
    Because the pencil is symmetric, left and right eigenvectors coincide and no
    eigenvector matrix is inverted.
    """
    r = len(omega)
    companion = np.block([[np.zeros((r, r)), np.eye(r)],
                          [-np.diag(omega**2), -Gamma]])
    lam, vec = la.eig(companion)
    phi = vec[:r]
    norm = (np.einsum('aj,ab,bj->j', phi, Gamma, phi, optimize=True)
            + 2*lam*np.einsum('aj,aj->j', phi, phi))
    # The normalized denominator is scale invariant for each eigenvector.
    # A small value signals self-orthogonality near a defective eigenvalue.
    denom_scale = (2*abs(lam)+la.norm(Gamma, 2))*np.sum(abs(phi)**2, axis=0)
    gap = abs(lam[:, None]-lam[None, :])
    np.fill_diagonal(gap, np.inf)
    pair_scale = np.maximum(1., np.maximum(abs(lam[:, None]), abs(lam[None, :])))
    if (np.min(abs(norm)/np.maximum(denom_scale, 1e-300)) < 1e-7
            or np.any(gap < 1e-8*pair_scale)):
        raise ArithmeticError('ill-conditioned or repeated QEP poles; use direct resolvent')
    return lam, phi/np.sqrt(norm)


@dataclass
class MechanicalModel:
    """Legacy Z stores T, or Z_R if evaluation supplies a reconstruction P."""

    Z: np.ndarray
    omega: np.ndarray
    Gamma: np.ndarray
    export_permutation: np.ndarray | None = None
    export_sign: np.ndarray | None = None
    optimization_bandwidth: int | None = None

    @property
    def r(self):
        return len(self.omega)

    def save(self, path):
        metadata = {}
        if self.export_permutation is not None:
            metadata.update(export_permutation=self.export_permutation,
                            export_sign=self.export_sign,
                            optimization_bandwidth=self.optimization_bandwidth)
        np.savez_compressed(path, Z=self.Z, omega=self.omega, Gamma=self.Gamma, **metadata)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z['Z'], z['omega'], z['Gamma'],
                   z['export_permutation'] if 'export_permutation' in z else None,
                   z['export_sign'] if 'export_sign' in z else None,
                   int(z['optimization_bandwidth']) if 'optimization_bandwidth' in z else None)

    def all_poles(self):
        """All 2r state eigenvalues, including real overdamped poles (s^-1)."""
        r = self.r
        ref = max(self.omega)
        companion = np.block([[np.zeros((r, r)), np.eye(r)],
                              [-np.diag((self.omega/ref)**2), -self.Gamma/ref]])
        return np.sort_complex(la.eigvals(companion)*ref)

    def poles(self):
        """Positive-imaginary representatives only; use all_poles for real poles."""
        lam = self.all_poles()
        return np.sort_complex(lam[lam.imag > 0])

    def state_space(self, S, B, P=None):
        z = self.Z if P is None else P@self.Z
        o = S@z
        b = z.T@B
        k = np.diag(self.omega**2)
        a = np.block([[np.zeros((self.r, self.r)), np.eye(self.r)], [-k, -self.Gamma]])
        return a, np.vstack([np.zeros_like(b), b]), np.c_[-o@k, -o@self.Gamma], o@b

    def evaluate(self, frequency, S, B, P=None):
        """Accelerance on the given frequencies, through the modal expansion."""
        z = self.Z if P is None else P@self.Z
        o = np.asarray(S@z)
        b = np.asarray(z.T@B)
        ref = max(self.omega)
        s = 2j*np.pi*np.atleast_1d(frequency)/ref
        try:
            lam, phi = modal_resolvent(self.omega/ref, self.Gamma/ref)
        except ArithmeticError:
            return np.array([v*v*o@la.solve(v*v*np.eye(self.r)+v*self.Gamma/ref
                                            +np.diag((self.omega/ref)**2), b)
                             for v in s])
        g = 1.0/(s[:, None]-lam[None, :])
        pred = (s[:, None, None]**2)*np.einsum('oj,fj,ji->foi', o@phi, g, phi.T@b,
                                              optimize=True)
        # Avoid subtractive cancellation between O(s) modal terms at infinity.
        high = abs(s) > 1e3*max(1., max(abs(lam)))
        if high.any():
            pred[high] = o@b+np.einsum('oj,fj,ji->foi', o@phi,
                                      g[high]*lam[None, :]**2, phi.T@b, optimize=True)
        return pred

    def checks(self, frequency, S, B, P=None):
        a, b, c, d = self.state_space(S, B, P)
        f = np.asarray(frequency)
        ss = np.array([c@la.solve(2j*np.pi*v*np.eye(2*self.r)-a, b)+d for v in f])
        z = self.Z if P is None else P@self.Z
        direct = np.array([(2j*np.pi*v)**2*(S@z)@la.solve(
            -(2*np.pi*v)**2*np.eye(self.r)+2j*np.pi*v*self.Gamma+np.diag(self.omega**2),
            z.T@B) for v in f])
        ev = la.eigvals(a)
        return dict(state_space_relative=float(la.norm(ss-direct)/max(la.norm(direct), 1e-30)),
                    evaluation_relative=float(la.norm(self.evaluate(f, S, B, P)-direct)
                                              / max(la.norm(direct), 1e-30)),
                    maximum_pole_real=float(ev.real.max()),
                    minimum_damping_eigenvalue=float(la.eigvalsh(self.Gamma)[0]),
                    positive_stiffness=bool(np.all(self.omega > 0)),
                    direct=d.tolist(), rank=rank_report(z))


# --------------------------------------------------------------------------
# objective
# --------------------------------------------------------------------------
def band_indices(r, width):
    """Lower-triangular index pairs within the optimization-coordinate bandwidth."""
    rows, cols = np.tril_indices(r)
    keep = (rows-cols) <= width
    return rows[keep], cols[keep]


class Objective:
    """Scaled complex accelerance misfit with analytic real-parameter gradients.

    Free parameters are the port coefficients W, the logarithms of the undamped
    frequencies, and a banded lower-triangular factor of the damping matrix.
    Bandwidth 0 is classical damping; bandwidth r-1 is the full dissipative
    family within the parameter bounds. Intermediate widths are a sparsity
    choice in optimization order, not a measured absence of distant coupling.
    """

    def __init__(self, f, y, S, B, E, Q, r, width, alpha=1e-5):
        self.f, self.y, self.r = f, y, r
        self.scale = scales(y)
        self.ref = 2*np.pi*max(f)
        self.s = 2j*np.pi*f/self.ref
        self.o = np.asarray(S@E)
        self.b = np.asarray(B.T@E)
        self.E = E
        self.p = E.shape[1]
        self.m, self.q_in = self.o.shape[0], self.b.shape[0]
        self.alpha = alpha
        self.width = int(min(width, r-1))
        self.tri = band_indices(r, self.width)
        q = E.T@(Q@E)
        q = (q+q.T)/2
        qnorm = la.norm(q, 2)
        self.q = q/qnorm if qnorm > 0 else np.zeros_like(q)
        self.pack_diagnostics = []
        self.zscale = np.sqrt(np.median(self.scale))
        self.n = y.size
        self.weight = 1.0/self.scale**2/self.n

    @property
    def damping_parameters(self):
        return len(self.tri[0])

    def pack(self, W, omega, Gamma):
        chol = la.cholesky(Gamma/self.ref-1e-7*np.eye(self.r), lower=True)
        projected = np.zeros_like(chol)
        projected[self.tri] = chol[self.tri]
        self.pack_diagnostics.append(dict(
            cholesky_projection_relative=float(la.norm(chol-projected)/la.norm(chol))))
        g = chol[self.tri].copy()
        g[self.tri[0] == self.tri[1]] = np.log(np.diag(chol))
        return np.r_[(W/self.zscale).ravel(), np.log(omega/self.ref), g]

    def unpack(self, x):
        nw = self.p*self.r
        W = x[:nw].reshape(self.p, self.r)*self.zscale
        w = np.exp(x[nw:nw+self.r])
        chol = np.zeros((self.r, self.r))
        chol[self.tri] = x[nw+self.r:]
        np.fill_diagonal(chol, np.exp(np.diag(chol)))
        return W, w, chol@chol.T+1e-7*np.eye(self.r), chol

    def transfer(self, x):
        W, w, gamma, _ = self.unpack(x)
        try:
            lam, phi = modal_resolvent(w, gamma)
        except ArithmeticError:
            return self._direct(x, gradient=False)
        o, b = self.o@W, self.b@W
        g = 1.0/(self.s[:, None]-lam[None, :])
        return (self.s[:, None, None]**2)*np.einsum('oj,fj,ji->foi', o@phi, g,
                                                    phi.T@b.T, optimize=True)

    def __call__(self, x):
        W, w, gamma, chol = self.unpack(x)
        try:
            lam, phi = modal_resolvent(w, gamma)
        except ArithmeticError:
            self.last_evaluation_method = 'direct'
            return self._direct(x)
        self.last_evaluation_method = 'modal'
        o, b = self.o@W, self.b@W
        ophi, cb, co = o@phi, phi.T@b.T, phi.T@o.T
        g = 1.0/(self.s[:, None]-lam[None, :])
        s2 = self.s**2
        pred = s2[:, None, None]*np.einsum('oj,fj,ji->foi', ophi, g, cb, optimize=True)
        err = pred-self.y
        cost = float(np.sum(abs(err/self.scale)**2)/self.n)
        adj = (err.conj()*self.weight).reshape(len(self.s), -1)

        gs2 = g*s2[:, None]
        psi2 = (gs2.T@adj).reshape(-1, self.m, self.q_in)
        psi3 = ((gs2*self.s[:, None]).T@adj).reshape(-1, self.m, self.q_in)
        rep2 = ((gs2*g).T@adj).reshape(-1, self.m, self.q_in)
        rep3 = ((gs2*g*self.s[:, None]).T@adj).reshape(-1, self.m, self.q_in)

        go = 2*np.real(phi@np.einsum('joi,ji->jo', psi2, cb, optimize=True)).T
        gb = 2*np.real(phi@np.einsum('joi,jo->ji', psi2, co, optimize=True)).T
        gW = self.o.T@go+self.b.T@gb

        t2 = self._divided(psi2, rep2, lam, co, cb, g, adj*s2[:, None])
        t3 = self._divided(psi3, rep3, lam, co, cb, g, adj*(s2*self.s)[:, None])
        gw = -2*np.real(np.einsum('aj,jk,ak->a', phi, t2, phi, optimize=True))*(2*w*w)
        gmat = -2*np.real(phi@t3@phi.T)
        gchol = (gmat+gmat.T)@chol
        gchol[np.diag_indices(self.r)] *= np.diag(chol)

        reg = self.alpha*np.sum(W*(self.q@W))/self.r/self.zscale**2
        gW += 2*self.alpha*(self.q@W)/self.r/self.zscale**2
        return cost+float(reg), np.r_[(gW*self.zscale).ravel(), gw, gchol[self.tri]]

    def _direct(self, x, gradient=True):
        """Dense resolvent and analytic gradient fallback; no eigenvector derivatives."""
        W, w, gamma, chol = self.unpack(x)
        o, b = self.o@W, self.b@W
        pred = []
        go, gb = np.zeros_like(o), np.zeros_like(b)
        gw, gmat = np.zeros_like(w), np.zeros_like(gamma)
        cost = 0.
        for k, s in enumerate(self.s):
            delta = s*s*np.eye(self.r)+s*gamma+np.diag(w*w)
            xb = la.solve(delta, b.T)
            h = s*s*o@xb
            pred.append(h)
            if gradient:
                xo = la.solve(delta, o.T)
                err = h-self.y[k]
                adj = err.conj()*self.weight
                cost += float(np.sum(abs(err/self.scale)**2)/self.n)
                go += 2*np.real(s*s*adj@xb.T)
                gb += 2*np.real(s*s*adj.T@xo.T)
                pair = xo@adj@xb.T
                gw -= 4*w*w*np.real(s*s*np.diag(pair))
                gmat -= 2*np.real(s**3*pair)
        if not gradient:
            return np.asarray(pred)
        gW = self.o.T@go+self.b.T@gb
        reg = self.alpha*np.sum(W*(self.q@W))/self.r/self.zscale**2
        gW += 2*self.alpha*(self.q@W)/self.r/self.zscale**2
        gchol = (gmat+gmat.T)@chol
        gchol[np.diag_indices(self.r)] *= np.diag(chol)
        return cost+float(reg), np.r_[(gW*self.zscale).ravel(), gw, gchol[self.tri]]

    def _divided(self, psi, repeated, lam, co, cb, g, weighted):
        """Second-order frequency sums from first-order ones.

        sum_f u_f g_fj g_fk = (sum_f u_f g_fj - sum_f u_f g_fk)/(lam_j - lam_k),
        so the full (2r, 2r) object never requires a sum over frequency lines.
        """
        upper = np.einsum('jo,joi,ki->jk', co, psi, cb, optimize=True)
        lower = np.einsum('jo,koi,ki->jk', co, psi, cb, optimize=True)
        gap = lam[:, None]-lam[None, :]
        close = abs(gap) < 1e-5*np.maximum(1., np.maximum(abs(lam[:, None]), abs(lam[None, :])))
        out = (upper-lower)/np.where(close, 1.0, gap)
        diagonal = np.arange(len(lam))
        out[diagonal, diagonal] = np.einsum('jo,joi,ji->j', co, repeated, cb)
        np.fill_diagonal(close, False)
        # The confluent g_j^2 approximation is not exact for distinct close poles.
        for j, k in zip(*np.nonzero(close)):
            summed = ((g[:, j]*g[:, k])@weighted).reshape(self.m, self.q_in)
            out[j, k] = co[j]@summed@cb[k]
        return out

    def model(self, x):
        """Sort by frequency and fix the sign of each column."""
        W, w, g, _ = self.unpack(x)
        order = np.argsort(w)
        W, w, g = W[:, order], w[order], g[np.ix_(order, order)]
        sign = np.sign(W[np.argmax(abs(W), axis=0), np.arange(self.r)])
        sign[sign == 0] = 1
        W = W*sign
        g = g*sign[:, None]*sign[None]
        return MechanicalModel(self.E@W, w*self.ref, g*self.ref, order, sign,
                               self.width), W


# --------------------------------------------------------------------------
# data-only initialisation
# --------------------------------------------------------------------------
def denominator_poles(f, y, r, iterations=6, decimate=4):
    """Common denominator by pole relocation, from training FRFs only.

    This is the relaxed vector-fitting iteration of Gustavsen and Semlyen: a
    sequence of linear least-squares problems whose fixed point has the
    identified poles.  Only measured training data enter.
    """
    fs, ys = f[::decimate], y[::decimate]
    s = 2j*np.pi*fs
    nf = len(s)
    c = ys.reshape(nf, -1)
    w = 1.0/scales(y).reshape(-1)
    beta = 2*np.pi*np.linspace(fs[0], fs[-1], r)
    p = -beta*1e-2+1j*beta
    for _ in range(iterations):
        b = _pair_basis(s, p)
        n = 2*r
        blocks = []
        for j in range(c.shape[1]):
            local = np.c_[b, np.ones(nf)]*w[j]
            shared = np.c_[-b*c[:, j][:, None], -c[:, j]]*w[j]
            rr = la.qr(np.c_[np.r_[local.real, local.imag],
                             np.r_[shared.real, shared.imag]], mode='r')[0]
            blocks.append(rr[n+1:, n+1:])
        m = np.vstack(blocks)
        constraint = np.r_[np.real(np.sum(b, axis=0)), nf]
        x = la.lstsq(np.vstack([m, constraint]), np.r_[np.zeros(m.shape[0]), nf])[0]
        sigma, d = x[:n], x[-1]
        a = np.zeros((n, n))
        a[0::2, 0::2] = np.diag(p.real)
        a[0::2, 1::2] = np.diag(p.imag)
        a[1::2, 0::2] = np.diag(-p.imag)
        a[1::2, 1::2] = np.diag(p.real)
        root = np.zeros(n)
        root[0::2] = 2.0
        ev = la.eigvals(a-np.outer(root, sigma)/d)
        ev = np.where(ev.real > 0, -abs(ev.real)+1j*ev.imag, ev)
        upper = ev[ev.imag > 1e-9]
        if len(upper) < r:
            middle = 2*np.pi*fs[len(fs)//2]
            spare = ev[ev.imag <= 1e-9]
            spare = spare[np.argsort(spare.real)][:r-len(upper)]
            upper = np.r_[upper, -abs(spare.real)+1j*np.full(len(spare), middle)]
        p = upper[np.argsort(upper.imag)][:r]
    return p


def _pair_basis(s, p):
    """Real-coefficient conjugate-pair partial fractions, interleaved."""
    d = 1.0/(s[:, None]-np.c_[p, p.conj()].ravel()[None, :])
    b = np.empty((len(s), 2*len(p)), complex)
    b[:, 0::2] = d[:, 0::2]+d[:, 1::2]
    b[:, 1::2] = 1j*(d[:, 0::2]-d[:, 1::2])
    return b


def peak_poles(f, y, r):
    """Resonance frequencies and half-power widths of the training block."""
    scale = scales(y)
    energy = np.mean(abs(y/scale)**2, axis=(1, 2))
    df = np.median(np.diff(f))
    found = find_peaks(energy, distance=max(1, int(.003*len(f))))[0]
    found = found[np.argsort(energy[found])[::-1]]
    chosen = list(found[:r])
    for j in np.linspace(0, len(f)-1, r+2, dtype=int)[1:-1]:
        if len(chosen) < r and j not in chosen:
            chosen.append(int(j))
    chosen = np.array(sorted(chosen[:r]))
    if len(chosen) != r:
        raise ValueError('insufficient frequency samples for the requested order')
    with warnings.catch_warnings():
        # Flat or isolated samples give a zero width; the floor below covers them.
        warnings.simplefilter('ignore', PeakPropertyWarning)
        widths = peak_widths(energy, chosen, rel_height=.5)[0]
    half = np.maximum(widths*df, 2*df)
    omega = 2*np.pi*f[chosen]
    return -np.pi*half+1j*np.sqrt(np.maximum(omega**2-(np.pi*half)**2, (2*np.pi*df)**2))


def _numerators(f, y, omega, gamma, o_port, b_port):
    """Real rank-one modal numerators at fixed poles, then port coefficients.

    The numerator least squares carries the same channel scaling as the
    objective, so weak transfers are not swamped by strong ones.
    """
    scale = scales(y)
    s = 2j*np.pi*f
    design = (s[:, None]**2)/(s[:, None]**2+s[:, None]*gamma[None, :]+(omega**2)[None, :])
    target = (y/scale).reshape(len(f), -1)
    num = la.lstsq(np.r_[design.real, design.imag], np.r_[target.real, target.imag],
                   lapack_driver='gelsy')[0].reshape(len(omega), *y.shape[1:])*scale
    port = np.r_[o_port, b_port]
    W = np.zeros((o_port.shape[1], len(omega)))
    for j in range(len(omega)):
        u, sv, vh = la.svd(num[j], full_matrices=False)
        out, inp = u[:, 0]*np.sqrt(sv[0]), vh[0]*np.sqrt(sv[0])
        # When force and response ports coincide, both signs must be consistent,
        # so the rank-one factors cannot simply be read off the port rows.
        best = None
        for sign in (1., -1.):
            x = la.lstsq(port, np.r_[out, sign*inp])[0]
            sol = least_squares(lambda v: ((np.outer(o_port@v, b_port@v)-num[j])
                                           / scale).ravel(), x, max_nfev=100,
                                ftol=1e-10, gtol=1e-8, xtol=1e-10)
            if best is None or np.sum(sol.fun**2) < best[0]:
                best = (float(np.sum(sol.fun**2)), sol.x)
        W[:, j] = best[1]
    return W


def initialize(f, y, r, o_port, b_port, decimate=4):
    """Declared data-only starts: resonance peaks, and denominator relocation.

    Poles come either from the resonance peaks and half-power widths of the
    training block, or from the common-denominator relocation; the port
    coefficients then follow from real rank-one modal numerators.  No
    generating matrix, exact mode or previously fitted model is read.  The
    lower initial misfit does not predict the better optimum, so both starts
    are returned and fit() compares them after a short common probe.
    """
    out = []
    for name, poles in (('training peaks', peak_poles(f, y, r)),
                        ('common denominator',
                         denominator_poles(f, y, r, decimate=decimate))):
        omega, gamma = abs(poles), -2*poles.real
        out.append((name, _numerators(f, y, omega, gamma, o_port, b_port),
                    omega, np.diag(gamma)))
    return out


def _descend(obj, x, bounds, maxiter):
    return minimize(obj, x, jac=True, method='L-BFGS-B', bounds=bounds,
                    options=dict(maxiter=maxiter, ftol=1e-12, gtol=1e-8,
                                 maxls=40, maxcor=20))


def fit(f, y, S, B, E, Q, r, width, alpha=1e-5, start=None, maxiter=400):
    """Minimise the scaled misfit over the declared model family.

    With no warm start, both declared initialisations are advanced over a short
    common probe and the one with the lower training loss continues.  Only
    training observations take part in that comparison.
    """
    obj = Objective(f, y, S, B, E, Q, r, width, alpha)
    if start is None:
        candidates = initialize(f, y, r, obj.o, obj.b)
    else:
        W = la.lstsq(E, start.Z)[0]
        w, gamma = start.omega, start.Gamma
        if start.export_permutation is not None:
            # Restore optimizer coordinates before bandwidth continuation.
            inv = np.argsort(start.export_permutation)
            sign = start.export_sign
            W = (W*sign)[..., inv]
            w = w[inv]
            gamma = (gamma*sign[:, None]*sign[None, :])[np.ix_(inv, inv)]
        if start.r < r:
            extra = initialize(f, y-start.evaluate(f, S, B), r-start.r, obj.o, obj.b)
            candidates = [(f'warm start plus {name}', np.c_[W, added], np.r_[w, wa],
                           la.block_diag(gamma, ga))
                          for name, added, wa, ga in extra]
        elif start.r != r:
            raise ValueError('initial order exceeds candidate order')
        else:
            candidates = [('warm start', W, w, gamma)]
    n = obj.p*r
    bounds = [(None, None)]*n
    bounds += [(np.log(max(f[0]*.05, .001)/max(f)), np.log(3.))]*r
    bounds += [(np.log(1e-5), np.log(2.)) if a == b else (-2., 2.)
               for a, b in zip(*obj.tri)]
    t = time.perf_counter()
    probe = []
    for name, W, w, gamma in candidates:
        try:
            trial = _descend(obj, obj.pack(W, w, gamma), bounds,
                             max(100, maxiter//8) if len(candidates) > 1 else maxiter)
        except ArithmeticError:
            continue
        probe.append((float(trial.fun), name, trial))
    if not probe:
        raise ArithmeticError('no admissible initial model')
    probe.sort(key=lambda v: v[0])
    loss, chosen, result = probe[0]
    spent = sum(int(trial.nit) for _, _, trial in probe)
    if len(candidates) > 1:
        result = _descend(obj, result.x, bounds, maxiter)
        spent += int(result.nit)
    model, W = obj.model(result.x)
    report = dict(r=r, states=2*r, width=obj.width,
                  damping_parameters=obj.damping_parameters, alpha=alpha,
                  iterations=spent, success=bool(result.success),
                  message=str(result.message), loss=float(result.fun),
                  gradient_inf=float(np.max(abs(result.jac))),
                  seconds=time.perf_counter()-t,
                  training=score(model.evaluate(f, S, B), y, obj.scale),
                  rank=rank_report(model.Z),
                  initialization=chosen,
                  probe=[dict(start=name, loss=value) for value, name, _ in probe],
                  warm_start_projections=obj.pack_diagnostics,
                  bandwidth_scope='optimization coordinate order; exported factors permuted together',
                  omega_hz=(model.omega/(2*np.pi)).tolist())
    return model, report


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------
def selection_rule(predictions, selections, scale, labels, block_ids, seed=2907,
                   bootstrap=600, max_global=.20, max_channel=.30):
    """Smallest candidate within one selection standard error of the best.

    Whole-band acquisitions are the resampling units when at least three are
    supplied; with fewer records, paired frequency-block resampling is used and
    is explicitly weaker.  No split is made between neighbouring Fourier bins.
    """
    pred = np.asarray(predictions)
    sel = np.asarray(selections)
    sq = abs((pred[:, None]-sel[None])/scale)**2
    per = np.sqrt(np.mean(sq, axis=(1, 2)))
    errors = np.sqrt(np.mean(sq, axis=(1, 2, 3, 4)))
    best = int(np.argmin(errors))
    rng = np.random.default_rng(seed)
    unique = np.unique(block_ids)
    counts = np.array([np.sum(block_ids == k) for k in unique])
    units = np.array([np.mean(sq[:, :, block_ids == k], axis=2)
                      for k in unique]).transpose(1, 2, 0, 3, 4)
    draws, pc = [], []
    for _ in range(bootstrap):
        ri = rng.integers(sel.shape[0], size=sel.shape[0])
        bi = (np.arange(len(unique)) if sel.shape[0] >= 3
              else rng.integers(len(unique), size=len(unique)))
        val = units[:, ri][:, :, bi]
        channel = np.average(val, axis=2, weights=counts[bi]).mean(axis=1)
        draws.append(np.sqrt(channel.mean(axis=(1, 2))))
        pc.append(np.sqrt(channel))
    draws, pc = np.array(draws), np.array(pc)
    delta = float(np.std(draws[:, best], ddof=1))
    per_delta = np.std(pc[:, best], axis=0, ddof=1)
    rows = []
    for j, label in enumerate(labels):
        statistical = errors[j] <= errors[best]+delta
        safeguards = bool(np.all(per[j] <= per[best]+per_delta)
                          and errors[j] <= max_global and per[j].max() <= max_channel)
        rows.append(dict(**label, selection=float(errors[j]),
                         standard_error=float(np.std(draws[:, j], ddof=1)),
                         per_channel=per[j].tolist(),
                         within_uncertainty=bool(statistical), safeguards=safeguards,
                         accepted=bool(statistical and safeguards)))
    good = [a for a in rows if a['accepted']]
    chosen = min(good, key=lambda a: (a['r'], a['width'])) if good else None
    return dict(selected=chosen, best=rows[best], delta_stat=delta,
                per_channel_delta=per_delta.tolist(), candidates=rows,
                uncertainty='whole-acquisition bootstrap' if sel.shape[0] >= 3 else
                'paired frequency-block bootstrap conditional on inspected acquisitions',
                independent_selection_acquisitions=int(sel.shape[0]), blocks=len(unique),
                global_limit=max_global, per_channel_limit=max_channel,
                bootstrap_draws=int(bootstrap), bootstrap_seed=int(seed),
                paired_candidate_draws=True)


def spatial_candidates(model, Q, S, B, f, y, scale, mandatory, counts, rigid=None,
                       weights=None, reconstruction_limit=.05, frf_limit=.01,
                       condition_limit=1e4):
    """Retain scalar coordinates from the learned rows, preserving ports."""
    z = model.Z
    _, _, piv = la.qr(z.T/np.maximum(la.norm(z, axis=0)[:, None], 1e-30), pivoting=True)
    order = list(dict.fromkeys(list(mandatory)+piv.tolist()))
    rows, arrays = [], {}
    root = np.sqrt(weights)[:, None] if weights is not None else 1.0
    for count in sorted(set(max(len(mandatory), min(int(v), len(z))) for v in counts)):
        p, keep, drop, info = reconstruction(Q, order[:count], rigid)
        zr = z[keep]
        full = p@zr
        pred = MechanicalModel(zr, model.omega, model.Gamma).evaluate(f, S, B, p)
        relative = float(la.norm((full-z)*root)/max(la.norm(z*root), 1e-30))
        rank = rank_report(zr)
        change = score(pred, model.evaluate(f, S, B), scale)['complex']
        info.update(selection=score(pred, y, scale), shape_reconstruction=relative,
                    rank=rank, frf_change=change,
                    passes=bool(rank['full_column_rank']
                                and rank['condition'] <= condition_limit
                                and relative <= reconstruction_limit
                                and change <= frf_limit))
        rows.append(info)
        arrays[count] = (p, keep, drop, zr)
    valid = [v['k'] for v in rows if v['passes']]
    return rows, arrays, min(valid) if valid else None
