"""Level metrics and declared ranking for exploratory spatial holdout studies."""
import numpy as np


def response_levels(frequency, measured, predicted):
    frequency = np.asarray(frequency)
    measured, predicted = np.asarray(measured), np.asarray(predicted)
    if (frequency.ndim != 1 or len(frequency) < 2 or
            not np.all(np.diff(frequency) > 0) or measured.shape != frequency.shape or
            predicted.shape != frequency.shape or not np.isfinite(measured).all() or
            not np.isfinite(predicted).all() or np.any(measured <= 0) or np.any(predicted <= 0)):
        raise ValueError('positive finite PSDs on a shared increasing frequency grid required')
    rms_measured = float(np.sqrt(np.trapezoid(measured, frequency)))
    rms_predicted = float(np.sqrt(np.trapezoid(predicted, frequency)))
    ratio = rms_predicted / rms_measured
    difference_db = 10 * np.log10(predicted / measured)
    return dict(measured_rms=rms_measured, predicted_rms=rms_predicted, rms_ratio=ratio,
                rms_level_db=float(20 * np.log10(ratio)),
                median_absolute_psd_db=float(np.median(abs(difference_db))),
                mean_absolute_psd_db=float(np.mean(abs(difference_db))),
                p90_absolute_psd_db=float(np.quantile(abs(difference_db), .9)))


def rank_trial(rows):
    """A symmetric penalty for amplitude under/overprediction, without rescaling."""
    if not rows:
        raise ValueError('at least one candidate channel is required')
    return (max(abs(r['rms_level_db']) for r in rows),
            float(np.mean([r['mean_absolute_psd_db'] for r in rows])),
            float(np.mean([r['complex_error'] for r in rows])))
