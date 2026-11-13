"""Multiple-input spectra and local polynomial FRFs, with explicit averaging.

All transforms use exp(-i omega t); responses have the exp(+i omega t)
phasor convention. No zero padding or independent-input approximation.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as la
from scipy.signal import get_window


def welch(a, f, rate=8192., duration=16., window='hann', overlap=.5,
          band=(100., 1000.)):
    n = int(round(rate*duration))
    step = int(round(n*(1-overlap)))
    if n > len(a) or n < 8 or step < 1:
        raise ValueError('invalid segment duration or overlap')
    win = get_window(window, n, fftbins=True)
    nu = np.fft.rfftfreq(n, 1/rate)
    keep = (nu >= band[0]) & (nu <= band[1])
    aa, ff = [], []
    for start in range(0, len(a)-n+1, step):
        ax, fx = a[start:start+n], f[start:start+n]
        aa.append(np.fft.rfft((ax-ax.mean(0))*win[:, None], axis=0)[keep])
        ff.append(np.fft.rfft((fx-fx.mean(0))*win[:, None], axis=0)[keep])
    A, F = np.array(aa), np.array(ff)
    scale = 2/(rate*(win@win))
    saf = np.einsum('kfo,kfi->foi', A, F.conj())*scale/len(A)
    sff = np.einsum('kfi,kfj->fij', F, F.conj())*scale/len(A)
    saa = np.mean(abs(A)**2, axis=0)*scale
    count = len(A)
    # Equivalent count under stationary Gaussian independent Fourier blocks;
    # NOT a claim that these overlapping segments are independent records.
    correction = 1.
    for lag in range(1, count):
        shift = lag*step
        if shift >= n:
            break
        corr = (win[:-shift]@win[shift:])/(win@win)
        correction += 2*(1-lag/count)*corr**2
    info = dict(duration_s=duration, spacing_hz=rate/n, window=str(window),
                overlap=overlap, segment_count=count,
                equivalent_averages_gaussian=count/correction,
                enbw_hz=rate*(win@win)/win.sum()**2,
                records=1, independent_records=1, zero_padding=False)
    return nu[keep], saf, sff, saa, info


def h1(saf, sff, saa):
    condition = np.linalg.cond(sff)
    H = np.linalg.solve(sff.swapaxes(-1, -2),
                           saf.swapaxes(-1, -2)).swapaxes(-1, -2)
    coherent = np.einsum('foi,fij,foj->fo', H, sff, H.conj()).real
    coherence = np.clip(coherent/np.maximum(saa, 1e-300), 0, 1)
    return H, coherence, condition


def pooled(blocks):
    counts = np.array([b[4]['segment_count'] for b in blocks])
    if any(not np.array_equal(b[0], blocks[0][0]) for b in blocks):
        raise ValueError('incompatible grids')
    spectra = [sum(b[i]*c for b, c in zip(blocks, counts))/sum(counts)
               for i in (1, 2, 3)]
    info = dict(blocks[0][4])
    info.update(segment_count=int(sum(counts)), records=len(blocks),
                independent_records=len(blocks),
                equivalent_averages_gaussian=sum(b[4]['equivalent_averages_gaussian']
                                                for b in blocks))
    return blocks[0][0], *h1(*spectra), info, spectra


def local_polynomial(records, rate=8192., band=(100., 1000.),
                     halfwidth=8, degree=2, stride=3):
    """Common MIMO FRF polynomial; a separate transient per full record.

    Y_r(k+l) = sum_i sum_d H_id(k) (l/n)^d F_ri(k+l)
               + sum_d T_rd(k) (l/n)^d + noise.
    Full unwindowed DFTs, degree-2 local fit by scaled QR/SVD. Coefficient
    variances are conditional on locally white output errors and exact force;
    independent-record repeatability must also be assessed. The estimator
    approximates the FRF over 2*halfwidth/(record duration) Hz.
    """
    n = len(records[0][0])
    if any(len(a) != n or len(f) != n for a, f in records):
        raise ValueError('common full-record length required')
    nf = records[0][1].shape[1]
    no = records[0][0].shape[1]
    A = [np.fft.rfft(a-a.mean(0), axis=0) for a, _ in records]
    F = [np.fft.rfft(f-f.mean(0), axis=0) for _, f in records]
    freq = np.fft.rfftfreq(n, 1/rate)
    centers = np.flatnonzero((freq >= band[0]) & (freq <= band[1]))[::stride]
    centers = centers[(centers >= halfwidth) & (centers < len(freq)-halfwidth)]
    offset = np.arange(-halfwidth, halfwidth+1)
    P = (offset[:, None]/halfwidth)**np.arange(degree+1)[None]
    q = degree+1
    nrows = len(P)*len(records)
    ncols = q*(nf+len(records))
    if nrows <= ncols:
        raise ValueError('insufficient local lines for transient and FRF terms')
    H = np.empty((len(centers), no, nf), complex)
    variance = np.empty(H.shape)
    condition = np.empty(len(centers))
    for j, k in enumerate(centers):
        X = np.zeros((nrows, ncols), complex)
        Y = np.concatenate([a[k+offset] for a in A])
        for r, f in enumerate(F):
            rows = slice(r*len(P), (r+1)*len(P))
            X[rows, :nf*q] = (f[k+offset, :, None]*P[:, None, :]).reshape(len(P), nf*q)
            X[rows, (nf+r)*q:(nf+r+1)*q] = P
        norm = la.norm(X, axis=0)
        X /= norm
        coef, _, rank, sv = la.lstsq(X, Y, lapack_driver='gelsd')
        condition[j] = sv[0]/max(sv[-1], 1e-300)
        mse = np.sum(abs(Y-X@coef)**2, axis=0)/(nrows-rank)
        inv = la.pinv(X)
        coeff_var = np.sum(abs(inv)**2, axis=1)/norm**2
        H[j] = (coef[:nf*q:q]/norm[:nf*q:q, None]).T
        variance[j] = mse[:, None]*coeff_var[None, :nf*q:q]
    return freq[centers], H, variance, condition, dict(
        estimator='local polynomial with record-specific transient', degree=degree,
        halfwidth_lines=halfwidth, fitting_width_hz=2*halfwidth*rate/n,
        dft_spacing_hz=rate/n, output_spacing_hz=stride*rate/n,
        duration_s=n/rate, records=len(records), independent_records=len(records),
        window='rectangular; explicit transient', overlap=0., zero_padding=False,
        variance_assumptions='locally white output errors; exact input; polynomial FRF')
