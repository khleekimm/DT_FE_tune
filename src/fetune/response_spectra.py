"""Measured-input response autospectra with explicit spectral conventions."""
import numpy as np
from scipy.signal import get_window


def spectral_matrix(samples, rate, nperseg, noverlap, window='hann'):
    """One-sided Welch matrix G_ij = mean(X_i conj(X_j))/(Hz normalization).

    Real sample rows, channel columns; segment mean removal, periodic window,
    arithmetic averaging, no padding. DC and Nyquist are not doubled.
    This independent estimator supports the new response-spectrum assessment;
    the saved local-polynomial FRFs and historical estimates are unchanged.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or not np.isfinite(samples).all():
        raise ValueError('finite real sample-by-channel array required')
    if not (rate > 0 and 8 <= nperseg <= len(samples) and 0 <= noverlap < nperseg):
        raise ValueError('invalid spectral estimator settings')
    step = nperseg-noverlap
    win = get_window(window, nperseg, fftbins=True)
    frequency = np.fft.rfftfreq(nperseg, 1/rate)
    matrix = np.zeros((len(frequency), samples.shape[1], samples.shape[1]), complex)
    starts = range(0, len(samples)-nperseg+1, step)
    for start in starts:
        segment = samples[start:start+nperseg]
        transform = np.fft.rfft((segment-segment.mean(0))*win[:, None], axis=0)
        matrix += transform[:, :, None]*transform[:, None, :].conj()
    one_sided = np.full(len(frequency), 2.)
    one_sided[0] = 1.
    if nperseg % 2 == 0:
        one_sided[-1] = 1.
    matrix *= (one_sided/(len(starts)*rate*np.dot(win, win)))[:, None, None]
    return frequency, matrix, dict(rate_hz=rate, nperseg=nperseg,
        noverlap=noverlap, segment_count=len(starts), window=str(window),
        periodic_window=True, detrend='segment mean', averaging='arithmetic mean',
        one_sided=True, normalization='density per Hz; DC/Nyquist not doubled',
        spacing_hz=rate/nperseg, enbw_hz=rate*np.dot(win,win)/win.sum()**2,
        zero_padding=False, used_samples=starts[-1]+nperseg,
        unused_tail_samples=len(samples)-starts[-1]-nperseg)


def propagate_spectra(transfer, force_spectra):
    """H G_ff H*, retaining complex transfer and correlated-input cross terms."""
    return np.einsum('foi,fij,fpj->fop', transfer, force_spectra,
                     transfer.conj(), optimize=True)


def psd_level(psd, reference=1.):
    """Power dB, with reference in the same PSD units; never conceal negatives."""
    psd = np.asarray(psd)
    if reference <= 0 or np.iscomplexobj(psd) or not np.isfinite(psd).all() or np.any(psd < 0):
        raise ValueError('PSD must be finite, real and nonnegative; positive reference required')
    with np.errstate(divide='ignore'):
        return 10*np.log10(psd/reference)

