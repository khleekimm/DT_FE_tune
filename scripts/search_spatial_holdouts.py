"""Refit separate spatial holdouts without using the accepted 31-response fit.

Run `prepare`, `trial --name NAME` for every declared trial, then `report`.
Every artifact goes to a separate directory; the accepted model and paper are
read-only. Candidate selection is exploratory and precedes record-102 access.
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
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fetune.io import digest, json_write
from fetune.mechanical import MechanicalModel, Objective, completion, score, scales
from fetune.structured_fit import kernels, conditional_solve, balance_scale
from fetune.fe_completion import (ElasticContext, energy_dictionary, project_operators,
                                 energy_coefficients, constrained_energy_coefficients,
                                 relative)
from fetune.completion_diagnostics import physical_diagnostics

CONFIG = ROOT / 'configs/spatial_holdout_search.json'
BASE = ROOT / 'artifacts/model/laboratory_expanded'
FE = ROOT / 'artifacts/fe_completion'


def read(path):
    return json.loads(Path(path).read_text())


def partition(channels, held, seed_channels):
    channels = np.asarray(channels)
    if len(set(held)) != len(held) or not set(held).issubset(channels):
        raise ValueError('holdout channels must be distinct available responses')
    if set(seed_channels) & set(held):
        raise ValueError('held responses were used in the initialization fit')
    fit = np.flatnonzero(~np.isin(channels, held))
    test = np.flatnonzero(np.isin(channels, held))
    if not set(seed_channels).issubset(channels[fit]):
        raise ValueError('initialization fit channels must remain in training')
    return fit, test


def prepare(cfg, out):
    out.mkdir(parents=True, exist_ok=True)
    old = ROOT / cfg['seed_directory']
    seed_obs = np.load(old / 'observations.npz')
    seed_channels = seed_obs['channels'][seed_obs['fit_rows']].tolist()
    assert seed_channels == cfg['seed_fit_channels']
    prior = read(old / 'prior.json')
    assert prior['selected'] == 'gradient_rigid'
    assert digest(old / 'Q.npz') == digest(BASE / 'Q.npz')
    obs = np.load(BASE / 'observations.npz')
    for trial in cfg['trials']:
        partition(obs['channels'], trial['held_channels'], seed_channels)
        assert set(trial['candidate_channels']).issubset(trial['held_channels'])
    seed = out / 'seed'
    seed.mkdir(exist_ok=True)
    for name in ['r384_w24.npz', 'r384_w24.json', 'prior.json', 'Q.npz']:
        target = seed / name
        if target.exists():
            assert digest(target) == digest(old / name)
        else:
            shutil.copyfile(old / name, target)
    # These are geometry and training-role metadata, without held measurements.
    np.savez_compressed(seed / 'registration.npz', E=seed_obs['E'],
                        fit_channels=np.array(seed_channels), S=seed_obs['S'], B=seed_obs['B'])
    protected = [BASE / 'observations.npz', BASE / 'Q.npz',
                 ROOT / 'artifacts/robust_revision/final_polish.npz',
                 ROOT / 'artifacts/fe_primary_revision/laboratory_completion.npz',
                 ROOT / 'artifacts/fe_primary_revision/laboratory_predictions.npz',
                 ROOT / 'artifacts/spectral_assessment/spectra.npz',
                 ROOT / 'paper/main.pdf', ROOT / 'paper/text/laboratory.tex']
    sources = [FE / 'M_nominal.npz', FE / 'K_nominal.npz', FE / 'dof_mapping.npz',
               ROOT / 'src/fetune/mechanical.py', ROOT / 'src/fetune/structured_fit.py',
               ROOT / 'src/fetune/fe_completion.py', CONFIG]
    declaration = dict(config=cfg, config_sha256=digest(CONFIG), seed_fit_channels=seed_channels,
        seed_sha256=digest(seed / 'r384_w24.npz'), seed_source=str(old),
        protected={str(p.relative_to(ROOT)): digest(p) for p in protected},
        sources={str(p.relative_to(ROOT)): digest(p) for p in sources},
        prior_selection=prior, baseline_role='comparison only; never used in a trial fit')
    target = out / 'declaration.json'
    if target.exists():
        previous = read(target)
        assert previous['config_sha256'] == declaration['config_sha256']
        assert previous['protected'] == declaration['protected']
    else:
        json_write(target, declaration)
    print('Prepared three declared trials; seed channels:', seed_channels, flush=True)


def solve_output(obj, x):
    W, w, gamma, _ = obj.unpack(x)
    U, info = conditional_solve(kernels(obj.s, w, gamma, W[obj.m:]),
        obj.y, obj.scale, obj.q, W, np.arange(obj.m), obj.alpha / obj.r / obj.zscale**2)
    if info['stationarity_relative'] > 1e-9:
        raise ArithmeticError('inexact conditional output solve')
    result = x.copy()
    result[:obj.m * obj.r] = (U / obj.zscale).ravel()
    return result, info


class BudgetExpired(Exception):
    pass


def fit_trial(cfg, out, trial):
    dest = out / trial['name']
    dest.mkdir(exist_ok=True)
    if (dest / 'fit.json').exists():
        assert read(dest / 'fit.json')['config_sha256'] == digest(CONFIG)
        print(trial['name'], 'reusing completed fit', flush=True)
        return
    obs = np.load(BASE / 'observations.npz')
    fit_rows, held_rows = partition(obs['channels'], trial['held_channels'], cfg['seed_fit_channels'])
    S, B = obs['S_all'][fit_rows], obs['B']
    A = np.vstack([S, B.T])
    Q = sp.load_npz(out / 'seed/Q.npz')
    E, einfo = completion(Q, A, obs['rigid'])
    assert einfo['port_rank'] == len(A)
    y = obs[cfg['training_record']][:, fit_rows]
    obj = Objective(obs['frequency'], y, S, B, E, Q, cfg['order'],
                    cfg['bandwidth'], cfg['alpha'])
    assert np.allclose(obj.o, np.eye(len(A))[:len(S)], atol=1e-12)
    assert np.allclose(obj.b, np.eye(len(A))[len(S):], atol=1e-12)
    seed = MechanicalModel.load(out / 'seed/r384_w24.npz')
    W, omega, gamma = A @ seed.Z, seed.omega, seed.Gamma
    if seed.export_permutation is not None:
        inv, sign = np.argsort(seed.export_permutation), seed.export_sign
        W = (W * sign)[:, inv]
        omega = omega[inv]
        gamma = (gamma * sign[:, None] * sign[None, :])[np.ix_(inv, inv)]
    x = obj.pack(W, omega, gamma)
    projection = dict(obj.pack_diagnostics[-1])
    projection['seed_has_optimizer_permutation'] = seed.export_permutation is not None
    if seed.export_permutation is not None:
        assert projection['cholesky_projection_relative'] < 1e-12
    else:
        # The original archive predates optimizer-order metadata. Record the
        # deterministic projection into the declared bandwidth before refitting.
        projection['rule'] = 'Archive lacks optimizer permutation; project Cholesky factor to bandwidth 24 in stored order, then refit.'
    x, info = solve_output(obj, x)
    initial, _ = obj(x)
    best, bestvalue = x.copy(), initial
    history = [dict(seconds=0., objective=initial)]
    nout, nw = obj.m * obj.r, obj.p * obj.r
    bounds = [(None, None)] * (nw - nout)
    bounds += [(np.log(max(obj.f[0] * .05, .001) / max(obj.f)), np.log(3.))] * obj.r
    bounds += [(np.log(1e-5), np.log(2.)) if a == b else (-2., 2.) for a, b in zip(*obj.tri)]
    start = time.perf_counter()
    last_print = start
    solves = [info]
    evaluations = 0

    def objective(v):
        nonlocal best, bestvalue, last_print, evaluations
        if time.perf_counter() - start >= cfg['optimization_seconds']:
            raise BudgetExpired()
        candidate = x.copy()
        candidate[nout:] = v
        candidate, check = solve_output(obj, candidate)
        value, gradient = obj(candidate)
        solves.append(check)
        evaluations += 1
        if value < bestvalue:
            best, bestvalue = candidate.copy(), float(value)
            history.append(dict(seconds=time.perf_counter() - start, objective=bestvalue))
        if time.perf_counter() - last_print > 30:
            print(trial['name'], 'fit', round(time.perf_counter() - start), 's; objective', bestvalue, flush=True)
            last_print = time.perf_counter()
        return value, gradient[nout:]

    termination = 'fixed wall-time budget'
    try:
        result = minimize(objective, x[nout:], jac=True, method='L-BFGS-B', bounds=bounds,
                          options=dict(maxiter=10000, ftol=1e-12, gtol=1e-8, maxls=40, maxcor=20))
        termination = str(result.message)
    except BudgetExpired:
        pass
    # Fixed final polish; choose by training objective only.
    x = best.copy()
    for _ in range(3):
        W, _, _, _ = obj.unpack(x)
        balanced, _ = balance_scale(W, obj.q, obj.m)
        x[:nw] = (balanced / obj.zscale).ravel()
        x, _ = solve_output(obj, x)
    value, gradient = obj(x)
    assert value <= bestvalue + 1e-10
    W, w, g, _ = obj.unpack(x)
    np.savez_compressed(dest / 'fit.npz', W=W, omega=w * obj.ref, Gamma=g * obj.ref,
        A=A, E=E, scale=obj.scale, x=x, fit_rows=fit_rows, held_rows=held_rows,
        channels=obs['channels'], frequency=obs['frequency'])
    prediction = obj.transfer(x)
    report = dict(config_sha256=digest(CONFIG), trial=trial, seed_fit_channels=cfg['seed_fit_channels'],
        fit_channels=obs['channels'][fit_rows], held_channels=obs['channels'][held_rows],
        initial_objective=initial, final_objective=value, seconds=time.perf_counter() - start,
        objective_evaluations=evaluations, gradient_inf=float(la.norm(gradient, np.inf)),
        output_gradient_inf=float(la.norm(gradient[:nout], np.inf)), termination=termination,
        history=history, extension=einfo, seed_damping_projection=projection,
        max_inner_stationarity=max(v['stationarity_relative'] for v in solves),
        training=score(prediction, y, obj.scale),
        information_boundary='Only record 96 fit_rows and the five-channel seed enter fitting. No accepted 31-response factors are read.')
    json_write(dest / 'fit.json', report)
    print(trial['name'], 'fitted training complex error', report['training']['complex'], flush=True)


def complete_trial(cfg, out, trial):
    dest = out / trial['name']
    if (dest / 'completion.json').exists():
        assert read(dest / 'completion.json')['fit_sha256'] == digest(dest / 'fit.npz')
        print(trial['name'], 'reusing completed FE map', flush=True)
        return
    fit = np.load(dest / 'fit.npz')
    W, omega, gamma = fit['W'], fit['omega'], fit['Gamma']
    K, M = sp.load_npz(FE / 'K_nominal.npz'), sp.load_npz(FE / 'M_nominal.npz')
    represented = np.load(FE / 'dof_mapping.npz')['represented_rows']
    A = np.zeros((len(W), K.shape[0]))
    A[:, represented] = fit['A']
    context = ElasticContext(K, A)
    start = time.perf_counter()
    print(trial['name'], 'building new FE fields for its reduced anchor set', flush=True)
    E, N, dictionary = energy_dictionary(K, M, A, len(omega), represented=represented, context=context)
    ops, residual = project_operators(K, M, E, N, context=context)
    Z, criterion = energy_coefficients(W, omega, ops, eta=0.)
    base = E @ W
    delta = cfg['relative_singular_margin'] * la.svdvals(base[represented])[0]
    ls_min = float(la.svdvals((base + N @ Z)[represented])[-1])
    if ls_min < delta:
        # A deterministic feasible start uses no response data or earlier completion.
        warm = Z.copy()
        amplitude = delta / la.svdvals(N[represented])[-1]
        for _ in range(30):
            warm = Z + amplitude * np.eye(N.shape[1], len(omega))
            if la.svdvals((base + N @ warm)[represented])[-1] >= 1.01 * delta:
                break
            amplitude *= 2
        else:
            raise ArithmeticError('could not construct a feasible deterministic initialization')
        Z, constrained = constrained_energy_coefficients(W, omega, ops,
            np.c_[E[represented], N[represented]], 0., delta * 1.001, warm,
            max_iterations=cfg['completion_iterations'], tolerance=cfg['completion_tolerance'],
            acceptance_margin=delta, feasible_starts=(warm,), adaptive_penalty=True)
        criterion.update(constrained=constrained, warm_amplitude=float(amplitude))
    full = base + N @ Z
    diagnostic = physical_diagnostics(K, M, A, full, W, omega, gamma, represented, delta,
                                      solve=context.solve)
    assert diagnostic['margin_feasible']
    assert diagnostic['port_relative'] < 1e-10
    assert diagnostic['left_inverse_frobenius'] < 1e-9
    np.savez_compressed(dest / 'completion.npz', T=full[represented], W=W,
                        omega=omega, Gamma=gamma, A=fit['A'], Z=Z, represented_rows=represented)
    json_write(dest / 'completion.json', dict(trial=trial, fit_sha256=digest(dest / 'fit.npz'),
        seconds=time.perf_counter() - start, dictionary=dictionary, criterion=criterion,
        required_sigma_min=float(delta), equality_only_sigma_min=ls_min,
        projection_solve_residual=residual, diagnostics=diagnostic,
        rule='Same zero-ridge full-FE objective and finite margin; no held response enters this construction.'))
    print(trial['name'], 'completed; anchor error', diagnostic['port_relative'],
          'minimum singular value', diagnostic['spectrum']['minimum'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'trial', 'report'])
    parser.add_argument('--name')
    args = parser.parse_args()
    cfg = read(CONFIG)
    out = ROOT / cfg['output']
    if args.action == 'prepare':
        prepare(cfg, out)
    else:
        assert read(out / 'declaration.json')['config_sha256'] == digest(CONFIG)
        if args.action == 'trial':
            trial = next(t for t in cfg['trials'] if t['name'] == args.name)
            fit_trial(cfg, out, trial)
            complete_trial(cfg, out, trial)
        else:
            from report_spatial_holdouts import report
            report(cfg, out)


if __name__ == '__main__':
    main()
