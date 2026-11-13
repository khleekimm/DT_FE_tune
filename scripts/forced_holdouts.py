"""Spatial holdout trials completed with the forced criterion J_F.

Each trial keeps its saved fit (theta) from artifacts/spatial_holdout_search and
rebuilds only the completion, with the record-96 forces and the same margin rule
as the primary model. Evaluation and the declared record-101 ranking reuse
scripts/report_spatial_holdouts.py unchanged. No held response enters any
completion choice.

    python scripts/forced_holdouts.py complete hold_35_40
    python scripts/forced_holdouts.py report
"""
from pathlib import Path
import argparse
import json
import shutil
import sys
import time

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from fetune.io import digest, json_write
from fetune.fe_completion import (ElasticContext, energy_dictionary, project_operators,
                                  modal_response_weight, forced_coefficients,
                                  constrained_forced_coefficients, forced_objective,
                                  forced_objective_full)
from fetune.completion_diagnostics import physical_diagnostics
from fetune.holdout_assessment import rank_trial

CFG = json.loads((ROOT / 'configs/spatial_holdout_search.json').read_text())
FORCED = json.loads((ROOT / 'configs/forced_fe_completion.json').read_text())
OLD = ROOT / CFG['output']
OUT = ROOT / 'artifacts/spatial_holdout_forced'
FE = ROOT / 'artifacts/fe_completion'
SPECTRA = ROOT / FORCED['output'] / (FORCED['forced']['record'] + '_force_spectra.npz')


def trial_by_name(name):
    return next(t for t in CFG['trials'] if t['name'] == name)


def complete_trial(trial):
    dest = OUT / trial['name']
    dest.mkdir(parents=True, exist_ok=True)
    for name in ['fit.npz', 'fit.json']:
        if not (dest / name).exists():
            shutil.copy2(OLD / trial['name'] / name, dest / name)
    if (dest / 'completion.json').exists():
        print(trial['name'], 'completion exists', flush=True)
        return
    start_time = time.perf_counter()
    fit = np.load(dest / 'fit.npz')
    W, omega, gamma = fit['W'], fit['omega'], fit['Gamma']
    K, M = sp.load_npz(FE / 'K_nominal.npz'), sp.load_npz(FE / 'M_nominal.npz')
    represented = np.load(FE / 'dof_mapping.npz')['represented_rows']
    A = np.zeros((len(W), K.shape[0]))
    A[:, represented] = fit['A']
    context = ElasticContext(K, A)
    cache = dest / 'dictionary.npz'
    if cache.exists():
        z = np.load(cache)
        E, N, ops = z['E'], z['N'], [z['KP'], z['MP'], z['HP']]
        dictionary = json.loads((dest / 'dictionary.json').read_text())
    else:
        print(trial['name'], 'building FE fields for its anchor set', flush=True)
        E, N, dictionary = energy_dictionary(K, M, A, len(omega), represented=represented, context=context)
        ops, residual = project_operators(K, M, E, N, context=context)
        dictionary['projection_solve_residual'] = residual
        np.savez_compressed(cache, E=E, N=N, KP=ops[0], MP=ops[1], HP=ops[2])
        json_write(dest / 'dictionary.json', dictionary)
    spectra = np.load(SPECTRA)
    Pi, weight = modal_response_weight(omega, gamma, W[-2:], spectra['frequency'], spectra['G_ff'])
    Z, criterion = forced_coefficients(W, omega, ops, Pi)
    Z_LS = Z.copy()
    base = E @ W
    delta = CFG['relative_singular_margin'] * la.svdvals(base[represented])[0]
    ls_min = float(la.svdvals((base + N @ Z)[represented])[-1])
    if ls_min < delta:
        rule = FORCED['rank_solver']
        h = W.shape[0]
        D = np.c_[E[represented], N[represented]]
        Rq = la.qr(D, mode='raw', check_finite=False)[1]
        RN, b = Rq[:, h:], Rq[:, :h] @ W
        u, s, vh = la.svd(b + RN @ Z, full_matrices=False)
        start = la.lstsq(RN, (u * np.maximum(s, 1.02 * delta)) @ vh - b)[0]
        Z, constrained = constrained_forced_coefficients(W, omega, ops, Pi, D, delta * rule['margin_guard'],
            start, rule['max_iterations'], rule['tolerance'], acceptance_margin=delta,
            feasible_starts=(start,), rho_scale=FORCED['forced']['rho_scale'],
            inner_tolerance=FORCED['forced']['inner_tolerance'], log_every=100)
        constrained['start_J_F'] = forced_objective(W, start, omega, ops, Pi)
        criterion['constrained'] = constrained
    full = base + N @ Z
    diagnostic = physical_diagnostics(K, M, A, full, W, omega, gamma, represented, delta, solve=context.solve)
    assert diagnostic['margin_feasible']
    assert diagnostic['port_relative'] < 1e-10
    assert diagnostic['left_inverse_frobenius'] < 1e-9
    criterion.update(J_F_accepted=forced_objective_full(K, M, full, omega, Pi, context.solve),
                     J_F_equality_only=forced_objective(W, Z_LS, omega, ops, Pi),
                     J0_equivalent_accepted=forced_objective(W, Z, omega, ops, np.diag(omega ** -2.)))
    np.savez_compressed(dest / 'completion.npz', T=full[represented], W=W, omega=omega, Gamma=gamma,
                        A=fit['A'], Z=Z, Z_LS=Z_LS, Pi=Pi, represented_rows=represented)
    json_write(dest / 'completion.json', dict(trial=trial, fit_sha256=digest(dest / 'fit.npz'),
        seconds=time.perf_counter() - start_time, dictionary=dictionary, criterion=criterion,
        modal_weight=weight, required_sigma_min=float(delta), equality_only_sigma_min=ls_min,
        diagnostics=diagnostic, force_record=FORCED['forced']['record'],
        rule='J_F with the record-96 forces, same margin rule as the primary completion; no held response enters.'))
    print(trial['name'], 'completed: J_F', criterion['J_F_accepted'], 'sigma_min',
          diagnostic['spectrum']['minimum'], 'delta', delta, flush=True)


def report():
    import report_spatial_holdouts as rsh
    for record in [CFG['screen_record'], CFG['assessment_record']]:
        for suffix in ['.npz', '.json']:
            target = OUT / (record + '_spectra' + suffix)
            if not target.exists():
                shutil.copy2(OLD / (record + '_spectra' + suffix), target)
    records = {rec: rsh.measured_spectra(OUT, rec) for rec in [CFG['screen_record'], CFG['assessment_record']]}
    results = [rsh.evaluate(OUT, trial, records, CFG) for trial in CFG['trials']]
    ranking = []
    for result in results:
        rows = [r for r in result['rows'] if r['record'] == CFG['screen_record'] and r['candidate']]
        ranking.append(dict(trial=result['trial']['name'], rank_key=rank_trial(rows),
                            channels=[r['channel'] for r in rows]))
    ranking.sort(key=lambda r: (r['rank_key'], r['trial']))
    previous = json.loads((OLD / 'selection.json').read_text())
    selection = dict(selected_trial=ranking[0]['trial'], selected_channels=ranking[0]['channels'],
        screen_record=CFG['screen_record'], rule=CFG['selection'], ranking=ranking,
        disclosure=CFG['scope'], criterion='J_F', previous_J0_selection=previous['selected_trial'],
        same_as_previous=ranking[0]['trial'] == previous['selected_trial'],
        candidates={r['trial']['name']: r['completion_sha256'] for r in results})
    json_write(OUT / 'selection.json', selection)
    summary = [dict(trial=r['trial']['name'], record=row['record'], channel=row['channel'],
                    complex_error=row['complex_error'], rms_ratio=row['rms_ratio'])
               for r in results for row in r['rows']]
    json_write(OUT / 'summary.json', dict(selection=selection, rows=summary))
    print('selected', selection['selected_trial'], 'same as J_0 selection:', selection['same_as_previous'])
    for row in summary:
        print(f"{row['trial']:15s} {row['record']:9s} ACC{row['channel']:<3d} complex {100 * row['complex_error']:8.2f}%  rms ratio {row['rms_ratio']:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['complete', 'report'])
    parser.add_argument('trial', nargs='?')
    args = parser.parse_args()
    if args.command == 'complete':
        complete_trial(trial_by_name(args.trial))
    else:
        report()
