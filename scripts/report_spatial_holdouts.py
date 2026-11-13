"""Assess every exploratory holdout, freeze selection, then check record 102."""
from pathlib import Path
import csv
import hashlib
import json
import sys

import h5py
import numpy as np
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fetune import data as source
from fetune.io import digest, json_write, savefig_atomic
from fetune.mechanical import MechanicalModel, score, scales
from fetune.fe_completion import relative
from fetune.response_spectra import spectral_matrix, propagate_spectra
from fetune.holdout_assessment import response_levels, rank_trial

BASE = ROOT / 'artifacts/model/laboratory_expanded'
POOL = np.array([8, 35, 38, 40])


def read(path):
    return json.loads(Path(path).read_text())


def scientific_tests(out):
    """Reuse this study's test record, or run the suite on a fresh reproduction."""
    path = out / 'scientific_tests.json'
    if path.exists():
        result = read(path)
        assert result['errors'] == result['failures'] == 0
        return result
    import importlib
    import inspect
    import unittest
    sys.path.insert(0, str(ROOT / 'tests'))
    suite, loader = unittest.TestSuite(), unittest.TestLoader()
    for test in sorted((ROOT / 'tests').glob('test_*.py')):
        module = importlib.import_module(test.stem)
        suite.addTests(loader.loadTestsFromModule(module))
        suite.addTests(unittest.FunctionTestCase(fn) for name, fn in inspect.getmembers(module, inspect.isfunction)
                       if name.startswith('test_') and fn.__module__ == test.stem)
    with (out / 'scientific_tests.log').open('w') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    assert result.wasSuccessful()
    summary = dict(run=result.testsRun, errors=len(result.errors), failures=len(result.failures))
    json_write(path, summary)
    return summary


def measured_spectra(out, record):
    cache = out / (record + '_spectra.npz')
    if cache.exists():
        assert read(cache.with_suffix('.json'))['cache_sha256'] == digest(cache)
        return dict(np.load(cache))
    raw = source.DATA / 'raw/august_2026/ship_mockup_timeseries_2026_08.h5'
    with h5py.File(raw, 'r') as handle:
        group = handle['measurement/' + record]
        assert np.array_equal(group.attrs['force_channels'], [59, 60])
        assert np.array_equal(group.attrs['excitation_positions'], [6, 8])
        force = group['force'][0]
        acceleration = np.concatenate([group['acceleration'][k][0]
                                       for k in ['deck', 'hull', 'fdn', 'added']], axis=1)[:, POOL - 1]
        attrs = {k: str(v) for k, v in group.attrs.items()}
    samples = np.column_stack([force, acceleration])
    frequency, G, info = spectral_matrix(samples, 8192., 131072, 65536)
    band = (frequency >= 360) & (frequency <= 1000)
    frequency, G = frequency[band], G[band]
    psd = np.diagonal(G[:, 2:, 2:], axis1=1, axis2=2).real
    ff, scipy_psd = welch(acceleration.astype(float), fs=8192., window='hann',
        nperseg=131072, noverlap=65536, detrend='constant', scaling='density', axis=0)
    assert np.array_equal(ff[band], frequency)
    check = relative(psd, scipy_psd[band])
    assert check < 1e-12
    result = dict(frequency=frequency, G_ff=G[:, :2, :2], measured_psd=psd, channels=POOL)
    np.savez_compressed(cache, **result)
    json_write(cache.with_suffix('.json'), dict(record=record, raw=str(raw),
        raw_size=raw.stat().st_size, raw_mtime_ns=raw.stat().st_mtime_ns,
        attrs=attrs, estimator=info, scipy_welch_relative=check, channels=POOL,
        array_sha256={k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
                      for k, v in [('force', force), ('acceleration', acceleration)]},
        cache_sha256=digest(cache), units='force N; acceleration m/s^2; one-sided PSD per Hz',
        extraction='Exact engineering-unit samples. No gains, signs, delays, or noise-floor corrections.'))
    return result


def evaluate(out, trial, records, cfg):
    dest = out / trial['name']
    obs = np.load(BASE / 'observations.npz')
    fit = np.load(dest / 'fit.npz')
    z = np.load(dest / 'completion.npz')
    model = MechanicalModel(z['T'], z['omega'], z['Gamma'])
    f, channels = obs['frequency'], obs['channels']
    prediction = model.evaluate(f, obs['S_all'], obs['B'])
    report_scale = scales(obs[cfg['training_record']])
    fit_rows, held_rows = fit['fit_rows'], fit['held_rows']
    ids = np.linspace(0, len(f) - 1, 41, dtype=int)
    h = len(z['W'])
    fixed = MechanicalModel(z['W'], z['omega'], z['Gamma']).evaluate(
        f[ids], np.eye(h)[:-2], np.eye(h)[:, -2:])
    invariance = relative(prediction[ids][:, fit_rows], fixed)
    assert invariance < 1e-10
    assert set(channels[held_rows]) == set(trial['held_channels'])
    assert not set(trial['held_channels']) & set(cfg['seed_fit_channels'])
    scores = {rec: score(prediction[:, fit_rows], obs[rec][:, fit_rows], fit['scale'])
              for rec in [cfg['training_record'], cfg['screen_record'], cfg['assessment_record']]}
    assert np.isclose(scores[cfg['training_record']]['complex'],
                      read(dest / 'fit.json')['training']['complex'], rtol=1e-10)
    pool_rows = [list(channels).index(c) for c in POOL]
    sf = next(iter(records.values()))['frequency']
    H = model.evaluate(sf, obs['S_all'][pool_rows], obs['B'])
    rows = []
    np.savez_compressed(dest / 'predictions.npz', frequency=f, prediction=prediction,
                        channels=channels, scale=report_scale, spectral_frequency=sf, spectral_H=H)
    for rec, measured in records.items():
        assert np.array_equal(sf, measured['frequency'])
        propagated = propagate_spectra(H, measured['G_ff'])
        psd = np.diagonal(propagated, axis1=1, axis2=2).real
        assert np.all(psd > 0)
        np.savez_compressed(dest / (rec + '_spectra.npz'), frequency=sf,
            measured_psd=measured['measured_psd'], predicted_psd=psd, channels=POOL)
        for channel in trial['held_channels']:
            j, k = list(channels).index(channel), list(POOL).index(channel)
            metric = response_levels(sf, measured['measured_psd'][:, k], psd[:, k])
            error = score(prediction[:, [j]], obs[rec][:, [j]], report_scale[[j]])
            rows.append(dict(trial=trial['name'], record=rec, channel=channel,
                candidate=channel in trial['candidate_channels'], **metric,
                complex_error=error['complex'], complex_P6=error['per_channel'][0][0],
                complex_P8=error['per_channel'][0][1],
                phase_P6_deg=error['phase_median_deg'][0][0],
                phase_P8_deg=error['phase_median_deg'][0][1]))
    result = dict(trial=trial, fitted_scores=scores, rows=rows,
        registered_frf_relative=invariance, fit_sha256=digest(dest / 'fit.npz'),
        completion_sha256=digest(dest / 'completion.npz'),
        disclosure=cfg['scope'])
    json_write(dest / 'assessment.json', result)
    return result


def confirm(out, trial, cfg):
    # The caller must have written the selection before requesting these samples.
    selection = read(out / 'selection.json')
    assert selection['selected_trial'] == trial['name']
    measured = measured_spectra(out, cfg['confirmation_record'])
    dest = out / trial['name']
    H = np.load(dest / 'predictions.npz')['spectral_H']
    psd = np.diagonal(propagate_spectra(H, measured['G_ff']), axis1=1, axis2=2).real
    rows = []
    for channel in trial['held_channels']:
        k = list(POOL).index(channel)
        rows.append(dict(trial=trial['name'], record=cfg['confirmation_record'], channel=channel,
            candidate=channel in trial['candidate_channels'],
            **response_levels(measured['frequency'], measured['measured_psd'][:, k], psd[:, k])))
    power = np.trapezoid(np.diagonal(measured['G_ff'], axis1=1, axis2=2).real,
                         measured['frequency'], axis=0)
    np.savez_compressed(dest / (cfg['confirmation_record'] + '_spectra.npz'),
        frequency=measured['frequency'], measured_psd=measured['measured_psd'], predicted_psd=psd, channels=POOL)
    result = dict(record=cfg['confirmation_record'], selected_trial=trial['name'], rows=rows,
        selection_sha256=digest(out / 'selection.json'), input_band_power=power,
        P8_over_P6_power_ratio=float(power[1] / power[0]),
        scope=cfg['confirmation'], selection_updated=False)
    json_write(out / 'confirmation.json', result)
    return result


def baseline_rows(out, cfg, records):
    """Report the original ACC8 failure beside the exploratory candidates."""
    obs = np.load(BASE / 'observations.npz')
    z = np.load(ROOT / 'artifacts/fe_primary_revision/laboratory_completion.npz')
    ids = [list(obs['channels']).index(c) for c in [8, 40]]
    model = MechanicalModel(z['T'], z['omega'], z['Gamma'])
    h = model.evaluate(records[cfg['screen_record']]['frequency'], obs['S_all'][ids], obs['B'])
    pred = np.load(ROOT / 'artifacts/fe_primary_revision/laboratory_predictions.npz')['prediction']
    scale = scales(obs[cfg['training_record']])
    rows = []
    for record, measured in records.items():
        psd = np.diagonal(propagate_spectra(h, measured['G_ff']), axis1=1, axis2=2).real
        for k, (ch, j) in enumerate(zip([8, 40], ids)):
            row = dict(trial='accepted_baseline', record=record, channel=ch, candidate=False,
                **response_levels(measured['frequency'], measured['measured_psd'][:, list(POOL).index(ch)], psd[:, k]),
                complex_error=score(pred[:, [j]], obs[record][:, [j]], scale[[j]])['complex'])
            rows.append(row)
        if record == 'test_103':
            existing = np.load(ROOT / 'artifacts/spectral_assessment/spectra.npz')
            assert relative(psd, existing['predicted_psd']) < 1e-10
            assert relative(measured['measured_psd'][:, [0, 3]], existing['measured_psd']) < 1e-12
    return rows


def figures(out, results, selection, confirmation, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from figure_layout import shared_legend

    plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
        'grid.alpha': .15, 'savefig.bbox': 'tight'})
    obs = np.load(BASE / 'observations.npz')
    measured = obs[cfg['assessment_record']]
    channels = obs['channels'].tolist()
    checks = []
    with PdfPages(out / 'all_trials.pdf') as document:
        for result in results:
            trial = result['trial']
            dest = out / trial['name']
            prediction = np.load(dest / 'predictions.npz')
            spectrum = np.load(dest / (cfg['assessment_record'] + '_spectra.npz'))
            candidates = trial['candidate_channels']
            fig, axes = plt.subplots(3, len(candidates), figsize=(3.8 * len(candidates), 8.2),
                                     layout='constrained', sharex=True, squeeze=False)
            magnitude_limits, psd_limits = [], []
            for col, ch in enumerate(candidates):
                j, k = channels.index(ch), list(POOL).index(ch)
                row_info = next(r for r in result['rows'] if r['record'] == cfg['assessment_record'] and r['channel'] == ch)
                for row, force in enumerate([0, 1]):
                    ax = axes[row, col]
                    for values, color, style, label in [
                            (abs(measured[:, j, force]), '.48', '-', 'Measured, record 103'),
                            (abs(prediction['prediction'][:, j, force]), '#0072B2', '--', 'FE completion, withheld sensor')]:
                        line, = ax.semilogy(prediction['frequency'], values, color=color, ls=style, lw=.65, label=label)
                        assert np.array_equal(line.get_ydata(), values)
                        magnitude_limits.extend([float(values.min()), float(values.max())])
                    ax.set_title(rf'ACC{ch} $\leftarrow$ P{6 + 2 * force}')
                    if col == 0:
                        ax.set_ylabel(r'Accelerance [m\,s$^{-2}$/N]')
                ax = axes[2, col]
                for key, color, style in [('measured_psd', '.48', '-'), ('predicted_psd', '#0072B2', '--')]:
                    values = spectrum[key][:, k]
                    line, = ax.semilogy(spectrum['frequency'], values, color=color, ls=style, lw=.65)
                    assert np.array_equal(line.get_ydata(), values)
                    psd_limits.extend([float(values.min()), float(values.max())])
                ax.set_title(rf'ACC{ch}: RMS ratio {row_info["rms_ratio"]:.2f}; {row_info["rms_level_db"]:+.1f} dB')
                ax.set_xlabel('Frequency [Hz]')
                if col == 0:
                    ax.set_ylabel(r'PSD [$(\mathrm{m\,s^{-2}})^2$/Hz]')
            for ax in axes[:2].flat:
                ax.set_ylim(min(magnitude_limits) / 1.2, max(magnitude_limits) * 1.2)
            for ax in axes[2]:
                ax.set_ylim(min(psd_limits) / 1.2, max(psd_limits) * 1.2)
            for ax in axes.flat:
                ax.set(xlim=(360, 1000), xticks=[400, 600, 800, 1000])
            shared_legend(fig, fontsize=10, ncol=2)
            savefig_atomic(fig, dest / 'comparison.pdf')
            savefig_atomic(fig, dest / 'comparison.png', dpi=180)
            document.savefig(fig, bbox_inches='tight')
            if trial['name'] == selection['selected_trial']:
                savefig_atomic(fig, out / 'recommended.pdf')
                savefig_atomic(fig, out / 'recommended.png', dpi=180)
            checks.append(dict(trial=trial['name'], channels=candidates,
                frf_samples=len(prediction['frequency']), spectral_samples=len(spectrum['frequency']),
                data='Exact saved magnitudes/PSDs; common limits across each figure; no gain, floor, clipping or smoothing.'))
            plt.close(fig)

    selected = next(t['trial'] for t in results if t['trial']['name'] == selection['selected_trial'])
    spectra = np.load(out / selected['name'] / (cfg['confirmation_record'] + '_spectra.npz'))
    fig, axes = plt.subplots(1, len(selected['candidate_channels']),
        figsize=(3.8 * len(selected['candidate_channels']), 3.5), layout='constrained', squeeze=False)
    for ax, ch in zip(axes.flat, selected['candidate_channels']):
        k = list(POOL).index(ch)
        row = next(r for r in confirmation['rows'] if r['channel'] == ch)
        for key, color, style, label in [('measured_psd', '.48', '-', 'Measured, record 102'),
                                       ('predicted_psd', '#0072B2', '--', 'Selected completed model')]:
            ax.semilogy(spectra['frequency'], spectra[key][:, k], color=color, ls=style, lw=.65, label=label)
        ax.set(title=rf'ACC{ch}: RMS ratio {row["rms_ratio"]:.2f}; {row["rms_level_db"]:+.1f} dB',
               xlabel='Frequency [Hz]', xlim=(360, 1000), xticks=[400, 600, 800, 1000])
    axes[0, 0].set_ylabel(r'PSD [$(\mathrm{m\,s^{-2}})^2$/Hz]')
    shared_legend(fig, fontsize=10, ncol=2)
    savefig_atomic(fig, out / 'confirmation.pdf')
    savefig_atomic(fig, out / 'confirmation.png', dpi=180)
    plt.close(fig)
    json_write(out / 'plot_checks.json', checks)


def report(cfg, out):
    records = {rec: measured_spectra(out, rec) for rec in [cfg['screen_record'], cfg['assessment_record']]}
    results = [evaluate(out, trial, records, cfg) for trial in cfg['trials']]
    ranking = []
    for result in results:
        rows = [r for r in result['rows'] if r['record'] == cfg['screen_record'] and r['candidate']]
        ranking.append(dict(trial=result['trial']['name'], rank_key=rank_trial(rows), channels=[r['channel'] for r in rows]))
    ranking.sort(key=lambda r: (r['rank_key'], r['trial']))
    selection = dict(selected_trial=ranking[0]['trial'], selected_channels=ranking[0]['channels'],
        screen_record=cfg['screen_record'], rule=cfg['selection'], ranking=ranking,
        disclosure=cfg['scope'], confirmation_record=cfg['confirmation_record'],
        confirmation_used_for_selection=False,
        candidates={r['trial']['name']: r['completion_sha256'] for r in results})
    target = out / 'selection.json'
    if target.exists():
        assert read(target) == json.loads(json.dumps(selection)), 'Selection is frozen; cannot change after confirmation.'
    else:
        json_write(target, selection)
    selected = next(t for t in cfg['trials'] if t['name'] == selection['selected_trial'])
    confirmation = confirm(out, selected, cfg)
    baseline = baseline_rows(out, cfg, records)
    all_rows = baseline + [r for result in results for r in result['rows']] + confirmation['rows']
    fields = list(dict.fromkeys(k for r in all_rows for k in r))
    with (out / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    figures(out, results, selection, confirmation, cfg)
    protected = read(out / 'declaration.json')['protected']
    for path, expected in protected.items():
        assert digest(ROOT / path) == expected, path
    checks = dict(protected_inputs_unchanged=len(protected),
        registered_frf_max_relative=max(r['registered_frf_relative'] for r in results),
        selection_before_confirmation=True, baseline_record103_spectra_replayed=True,
        primary_paper_unchanged=True,
        scientific_tests=scientific_tests(out),
        sources={str(p.relative_to(ROOT)): digest(p) for p in [
            Path(__file__), ROOT / 'scripts/search_spatial_holdouts.py',
            ROOT / 'src/fetune/holdout_assessment.py', ROOT / 'configs/spatial_holdout_search.json']})
    json_write(out / 'verification.json', checks)
    lines = ['EXPLORATORY SPATIAL HOLDOUT COMPARISON', '',
        'Selected trial: ' + selection['selected_trial'],
        'Candidate channels: ' + ', '.join('ACC' + str(c) for c in selection['selected_channels']),
        'Selection uses record 101 only; record 102 was read after selection was frozen.',
        cfg['scope'], '',
        'The original accepted model, manuscript, and ACC8/40 result remain unchanged.',
        'RMS ratio = predicted / measured; level difference = 20 log10(RMS ratio).',
        'Complex error uses record-96 RMS scales and includes phase; lower is better.', '',
        'Trial                Record     ACC  Measured RMS  Predicted RMS   Ratio  Level dB  Complex %']
    for r in all_rows:
        error = f'{100 * r["complex_error"]:9.2f}' if 'complex_error' in r else '      n/a'
        lines.append(f'{r["trial"]:20s} {r["record"]:10s} {r["channel"]:3d} '
            f'{r["measured_rms"]:13.5f} {r["predicted_rms"]:14.5f} {r["rms_ratio"]:7.3f} '
            f'{r["rms_level_db"]:9.3f} {error}')
    lines += ['', 'Fitting and numerical checks:']
    for result in results:
        name = result['trial']['name']
        fit = read(out / name / 'fit.json')
        completion = read(out / name / 'completion.json')
        solve = completion['criterion'].get('constrained', {})
        scores = result['fitted_scores']
        lines.append(f'{name}: train/monitor/development errors = ' +
            '/'.join(f'{100 * scores[rec]["complex"]:.2f}%' for rec in
                     [cfg['training_record'], cfg['screen_record'], cfg['assessment_record']]) +
            f'; fit evaluations {fit["objective_evaluations"]}, {fit["seconds"]:.1f} s.')
        lines.append(f'  FE rank = {completion["diagnostics"]["spectrum"]["rank"]}; '
            f'minimum singular value {completion["diagnostics"]["spectrum"]["minimum"]:.6g}, '
            f'margin {completion["required_sigma_min"]:.6g}; '
            f'solver termination: {solve.get("termination", "equality-only feasible")}.')
    lines += ['', 'Limitations:',
        '- Candidate channels are selected for level agreement; this is exploratory, not blind spatial validation.',
        '- The older seed excludes every trial holdout. ACC35/38 were used by the later primary fit, which is never used here.',
        '- The old seed lacks optimizer-order metadata: its Cholesky factor is projected to bandwidth 24 before the independent refits (5.15% relative factor change).',
        '- Each fit has a fixed 300 s search budget plus initialization/polish; nonlinear convergence is not claimed.',
        '- Each FE solve uses the fixed zero-ridge criterion, 2400-iteration cap and margin; feasibility does not certify stationarity.',
        '- All inherited geometry/direction uncertainties remain. ACC8 is retained in the table as a diagnostic.',
        f'- Record 102 P8/P6 force band-power ratio = {confirmation["P8_over_P6_power_ratio"]:.6g}; its confirmation assesses output levels with predominantly P6 excitation.',
        '- Full-band PSD/FRF arrays have no fitted amplitude correction, display floor, or additional smoothing.', '',
        'Figures: artifacts/spatial_holdout_search/{recommended,all_trials,confirmation}.pdf',
        'All scalar results: artifacts/spatial_holdout_search/comparison.csv',
        'Configuration: configs/spatial_holdout_search.json',
        'Reproduce: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/search_spatial_holdouts.py prepare',
        'Then: scripts/search_spatial_holdouts.py trial --name NAME for each configured name; finally report.',
        f'Preservation: {checks["protected_inputs_unchanged"]} protected inputs unchanged; maximum registered-FRF discrepancy {checks["registered_frf_max_relative"]:.3e}.',
        f'Tests: {checks["scientific_tests"]["run"]} passed, {checks["scientific_tests"]["failures"]} failures, {checks["scientific_tests"]["errors"]} errors.']
    text = '\n'.join(lines) + '\n'
    (out / 'report.txt').write_text(text)
    (ROOT / 'docs/spatial_holdout_search_report.txt').write_text(text)
    print(text, flush=True)
