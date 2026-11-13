"""Force estimation at known positions from measured accelerations and the model.

Two laboratory cases, both outside the fitting record:
  two forces : record 103, P6 and P8 random; both estimated from 31 accelerations.
  one force  : record 104, P6 random and P8 ramp; in 360--1000 Hz the ramp
               carries 0.04% of its power, so the band is a one-force case.
The estimator is generalized least squares per DFT line through H_theta. The
error model is the per-channel model-error spectrum of the training record 96
(residual power of the fit, averaged over both inputs and a frequency window);
the window is selected on the monitoring record 101 and then frozen. Records
103 and 104 are reported only. No regularization, gain or delay is fitted.
A compact cache makes the build self-contained; the raw archive is opened
read-only only when the cache is absent or --refresh is given.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fetune.io import digest,json_write
from fetune.mechanical import MechanicalModel
from fetune.response_spectra import spectral_matrix
OUT=ROOT/'artifacts/force_estimation'
MODEL=ROOT/'artifacts/fe_primary_revision/laboratory_completion.npz'
OBS=ROOT/'artifacts/model/laboratory_expanded/observations.npz'
PRED=ROOT/'artifacts/fe_primary_revision/laboratory_predictions.npz'
SETTINGS=ROOT/'artifacts/preserved/spectra/e3_support.json'
CASES={'two':dict(record='test_103',inputs=[0,1],signal='random + random'),
       'one':dict(record='test_104',inputs=[0],signal='random + ramp')}
SELECTION_RECORD='test_101';WINDOWS_HZ=[.25,.5,1.,2.,5.,10.,20.]
BAND=(360.,1000.);RATE=8192.;ASSESS=(6.,58.);DISPLAY=(20.,20.05)


def load_record(record):
    """Exact recorded forces and the 31 fitted accelerations, archive read-only."""
    import h5py
    from fetune import data as source
    raw=source.DATA/'raw/august_2026/ship_mockup_timeseries_2026_08.h5'
    obs=np.load(OBS);channels=obs['channels'][obs['fit_rows']]
    with h5py.File(raw,'r') as handle:
        group=handle['measurement/'+record]
        assert np.array_equal(group.attrs['force_channels'],[59,60])
        assert np.array_equal(group.attrs['excitation_positions'],[6,8])
        assert group.attrs['n_samples']==491520 and group.attrs['duration_s']==60
        force=group['force'][0].astype(np.float32)
        stacked=np.concatenate([group['acceleration'][k][0] for k in ['deck','hull','fdn','added']],axis=1)
        acceleration=stacked[:,channels-1].astype(np.float32)
        meta=dict(raw=str(raw),record='measurement/'+record,datetime=str(group.attrs['datetime']),
            signals=[str(group.attrs['exciter1_signal']),str(group.attrs['exciter2_signal'])],
            source_csv=str(group.attrs['source_csv']),channels=channels.tolist(),
            channel_order='concatenated deck, hull, fdn, added groups indexed by channel-1',
            array_sha256=dict(force=hashlib.sha256(np.ascontiguousarray(force).tobytes()).hexdigest(),
                acceleration=hashlib.sha256(np.ascontiguousarray(acceleration).tobytes()).hexdigest()))
    return force,acceleration,meta


def band_transform(x):
    """Mean removal, full-record real DFT and the inclusive 360--1000 Hz mask."""
    n=len(x);f=np.fft.rfftfreq(n,1/RATE);keep=(f>=BAND[0])&(f<=BAND[1])
    X=np.fft.rfft(np.asarray(x,float)-np.mean(x,axis=0),axis=0)
    return f,keep,X


def transfer(frequency,columns):
    z=np.load(MODEL);obs=np.load(OBS)
    model=MechanicalModel(z['T'],z['omega'],z['Gamma']);S=obs['S'];B=obs['B'][:,columns]
    H=np.empty((len(frequency),S.shape[0],len(columns)),complex)
    for start in range(0,len(frequency),4096):
        H[start:start+4096]=model.evaluate(frequency[start:start+4096],S,B)
    return H


def error_spectrum(frequency,window_hz):
    """Per-channel model-error RMS of the training fit, smoothed over a window.

    The residual of the fitted model on record 96 is averaged over both inputs
    and over the window on the 0.05 Hz reporting grid, then interpolated to the
    DFT lines. Edge windows are normalized by their actual sample count.
    """
    obs=np.load(OBS);pred=np.load(PRED);fit=obs['fit_rows'];grid=obs['frequency']
    power=np.mean(abs(pred['prediction'][:,fit]-obs['test_96'][:,fit])**2,axis=2)
    width=max(1,int(round(window_hz/np.median(np.diff(grid)))));kernel=np.ones(width)
    counts=np.convolve(np.ones(len(grid)),kernel,mode='same')
    smoothed=np.array([np.convolve(power[:,o],kernel,mode='same')/counts for o in range(power.shape[1])]).T
    return np.sqrt(np.array([np.interp(frequency,grid,smoothed[:,o]) for o in range(power.shape[1])]).T)


def solve(H,A,sigma):
    """Generalized least squares per line with diagonal error model sigma (lines x channels)."""
    Hw=H/sigma[...,None];Aw=A/sigma
    gram=np.einsum('foi,foj->fij',Hw.conj(),Hw);rhs=np.einsum('foi,fo->fi',Hw.conj(),Aw)
    F=np.linalg.solve(gram,rhs[...,None])[...,0]
    residual=Aw-np.einsum('foi,fi->fo',Hw,F)
    fit_fraction=1-np.sum(abs(residual)**2,axis=1)/np.sum(abs(Aw)**2,axis=1)
    return F,np.linalg.svd(Hw,compute_uv=False),fit_fraction


def waveforms(f,keep,F_measured,F_estimated,columns,n):
    Fm=np.zeros_like(F_measured);Fm[keep]=F_measured[keep]
    Fe=np.zeros((len(f),2),complex);Fe[np.ix_(keep,columns)]=F_estimated
    measured=np.fft.irfft(Fm,n=n,axis=0);estimated=np.fft.irfft(Fe,n=n,axis=0)
    t=np.arange(n)/RATE;window=(t>=ASSESS[0])&(t<=ASSESS[1])
    error=[float(np.sqrt(np.mean((estimated[window,i]-measured[window,i])**2)/np.mean(measured[window,i]**2))) for i in columns]
    correlation=[float(np.corrcoef(estimated[window,i],measured[window,i])[0,1]) for i in columns]
    return measured,estimated,t,window,error,correlation


def select_window(refresh=False):
    """Choose the error-model window on record 101 only; freeze it afterwards."""
    path=OUT/'selection.json'
    if path.exists() and not refresh:return json.loads(path.read_text())
    force,acceleration,meta=load_record(SELECTION_RECORD)
    f,keep,A=band_transform(acceleration);_,_,F=band_transform(force);nu=f[keep];H=transfer(nu,[0,1])
    rows=[]
    scale=np.load(PRED)['scale'][np.load(OBS)['fit_rows']]
    level=np.broadcast_to(np.sqrt(np.mean(scale**2,axis=1)),(len(nu),len(scale)))
    Fest,_,_=solve(H,A[keep],level);_,_,_,_,error,_=waveforms(f,keep,F,Fest,[0,1],len(force))
    rows.append(dict(weighting='signal level',window_hz=None,errors=error,mean=float(np.mean(error))))
    for window in WINDOWS_HZ:
        Fest,_,_=solve(H,A[keep],error_spectrum(nu,window));_,_,_,_,error,_=waveforms(f,keep,F,Fest,[0,1],len(force))
        rows.append(dict(weighting='training error spectrum',window_hz=window,errors=error,mean=float(np.mean(error))))
    candidates=[r for r in rows if r['window_hz'] is not None]
    best=min(candidates,key=lambda r:r['mean'])
    selection=dict(record='measurement/'+SELECTION_RECORD,metric='mean relative RMS error of P6 and P8 over 6--58 s',
        candidates=rows,selected_window_hz=best['window_hz'],selected_errors=best['errors'],
        rule='smallest mean error among the training-error-spectrum windows; the signal-level row is the reference only',
        record_sha256=meta['array_sha256'])
    OUT.mkdir(parents=True,exist_ok=True);json_write(path,selection)
    return selection


def estimate(case,window,refresh=False):
    spec=CASES[case];cache=OUT/(case+'_force.npz');meta_path=OUT/(case+'_record.json')
    if cache.exists() and meta_path.exists() and not refresh:
        return dict(np.load(cache)),json.loads(meta_path.read_text())
    force,acceleration,meta=load_record(spec['record'])
    f,keep,A=band_transform(acceleration);_,_,F=band_transform(force)
    nu=f[keep];H=transfer(nu,spec['inputs']);sigma=error_spectrum(nu,window);n=len(force)
    Fest,singular,fit_fraction=solve(H,A[keep],sigma)
    measured,estimated,t,window_mask,error,correlation=waveforms(f,keep,F,Fest,spec['inputs'],n)
    # Reference: the signal-level weighting, reported for comparison only.
    scale=np.load(PRED)['scale'][np.load(OBS)['fit_rows']]
    level=np.broadcast_to(np.sqrt(np.mean(scale**2,axis=1)),(len(nu),len(scale)))
    Fref,_,_=solve(H,A[keep],level);_,_,_,_,error_level,_=waveforms(f,keep,F,Fref,spec['inputs'],n)
    both={}
    if spec['inputs']!=[0,1]:
        # Control: with both columns admitted, model error leaks into a force
        # that is absent in the band; its size is reported in the text.
        Fb,_,_=solve(transfer(nu,[0,1]),A[keep],sigma)
        _,est2,_,_,error2,_=waveforms(f,keep,F,Fb,[0,1],n)
        both=dict(estimated_both=est2.astype(np.float32),
            band_rms_estimated_both=np.sqrt(np.mean(est2[window_mask]**2,axis=0)),relative_error_both=np.array(error2))
    archived=json.loads(SETTINGS.read_text())['settings']['hann16']
    nperseg=round(RATE*archived['duration_s']);noverlap=round(nperseg*archived['overlap'])
    wf,Gm,info=spectral_matrix(measured,RATE,nperseg,noverlap,archived['window'])
    _,Ge,_=spectral_matrix(estimated,RATE,nperseg,noverlap,archived['window'])
    inband=(wf>=BAND[0])&(wf<=BAND[1])
    psd_measured=np.diagonal(Gm,axis1=1,axis2=2).real[inband];psd_estimated=np.diagonal(Ge,axis1=1,axis2=2).real[inband]
    band_rms_measured=np.sqrt(np.mean(measured[window_mask]**2,axis=0));band_rms_estimated=np.sqrt(np.mean(estimated[window_mask]**2,axis=0))
    display=(t>=DISPLAY[0])&(t<=DISPLAY[1])
    arrays=dict(time=t[display].astype(np.float32),measured_display=measured[display].astype(np.float32),
        estimated_display=estimated[display].astype(np.float32),
        measured=measured.astype(np.float32),estimated=estimated.astype(np.float32),
        frequency=wf[inband],psd_measured=psd_measured,psd_estimated=psd_estimated,
        line_frequency=nu,singular_values=singular,fit_fraction=fit_fraction,
        inputs=np.array(spec['inputs']),error_window_hz=np.array(window),
        band_rms_measured=band_rms_measured,band_rms_estimated=band_rms_estimated,
        relative_error=np.array(error),correlation=np.array(correlation),
        relative_error_signal_level=np.array(error_level),**both)
    OUT.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache,**arrays)
    power=np.sum(abs(F[keep])**2,axis=0)/np.sum(abs(F)**2,axis=0);condition=singular[:,0]/singular[:,-1]
    meta.update(case=case,inputs_estimated=[f'P{6+2*i}' for i in spec['inputs']],signal_case=spec['signal'],
        band_hz=list(BAND),dft_lines=int(keep.sum()),spacing_hz=RATE/n,
        estimator='generalized least squares per DFT line; diagonal error model = training-record (96) model-error RMS per channel, averaged over both inputs and a frequency window selected on record 101; no regularization, gain, delay or smoothing',
        error_window_hz=window,assessment_window_s=list(ASSESS),display_window_s=list(DISPLAY),
        in_band_force_power_fraction=power.tolist(),
        band_rms_measured_N=band_rms_measured.tolist(),band_rms_estimated_N=band_rms_estimated.tolist(),
        relative_rms_error=error,correlation=correlation,relative_rms_error_signal_level=error_level,
        both_columns=dict(band_rms_estimated_N=both['band_rms_estimated_both'].tolist(),relative_rms_error=both['relative_error_both'].tolist()) if both else None,
        fit_fraction_median=float(np.median(fit_fraction)),
        condition_quantiles=dict(q50=float(np.median(condition)),q90=float(np.quantile(condition,.9)),max=float(condition.max())),
        welch=info,cache_sha256=digest(cache),
        sources={str(p.relative_to(ROOT)):digest(p) for p in [MODEL,OBS,PRED,SETTINGS,Path(__file__)]})
    json_write(meta_path,meta)
    return arrays,meta


def main(refresh=False):
    selection=select_window(refresh);window=selection['selected_window_hz']
    reports={case:estimate(case,window,refresh)[1] for case in CASES}
    two,one=reports['two'],reports['one']
    def macro(name,value):return r'\newcommand{\%s}{%s}'%(name,value)
    tex=[macro('ForceWindow',f'{window:g}'),
         macro('ForceSelectSix',f'{100*selection["selected_errors"][0]:.1f}'),macro('ForceSelectEight',f'{100*selection["selected_errors"][1]:.1f}'),
         macro('ForceSelectLevelSix',f'{100*selection["candidates"][0]["errors"][0]:.1f}'),macro('ForceSelectLevelEight',f'{100*selection["candidates"][0]["errors"][1]:.1f}'),
         macro('ForceTwoErrorSix',f'{100*two["relative_rms_error"][0]:.1f}'),macro('ForceTwoErrorEight',f'{100*two["relative_rms_error"][1]:.1f}'),
         macro('ForceTwoLevelSix',f'{100*two["relative_rms_error_signal_level"][0]:.1f}'),macro('ForceTwoLevelEight',f'{100*two["relative_rms_error_signal_level"][1]:.1f}'),
         macro('ForceTwoCorrSix',f'{two["correlation"][0]:.3f}'),macro('ForceTwoCorrEight',f'{two["correlation"][1]:.3f}'),
         macro('ForceTwoCondition',f'{two["condition_quantiles"]["q50"]:.1f}'),macro('ForceTwoConditionMax',f'{two["condition_quantiles"]["max"]:.1f}'),
         macro('ForceTwoFit',f'{100*two["fit_fraction_median"]:.1f}'),
         macro('ForceOneError',f'{100*one["relative_rms_error"][0]:.1f}'),macro('ForceOneLevel',f'{100*one["relative_rms_error_signal_level"][0]:.1f}'),
         macro('ForceOneCorr',f'{one["correlation"][0]:.3f}'),macro('ForceOneFit',f'{100*one["fit_fraction_median"]:.1f}'),
         macro('ForceOneRampFraction',f'{100*one["in_band_force_power_fraction"][1]:.2f}'),
         macro('ForceOneRampRms',f'{one["band_rms_measured_N"][1]:.3f}'),macro('ForceOneRandomRms',f'{one["band_rms_measured_N"][0]:.2f}'),
         macro('ForceTwoRmsSix',f'{two["band_rms_measured_N"][0]:.2f}'),macro('ForceTwoRmsEight',f'{two["band_rms_measured_N"][1]:.2f}'),
         macro('ForceTwoEstRmsSix',f'{two["band_rms_estimated_N"][0]:.2f}'),macro('ForceTwoEstRmsEight',f'{two["band_rms_estimated_N"][1]:.2f}'),
         macro('ForceOneEstRms',f'{one["band_rms_estimated_N"][0]:.2f}'),
         macro('ForceOneBothSix',f'{100*one["both_columns"]["relative_rms_error"][0]:.1f}'),
         macro('ForceOneBothEightRms',f'{one["both_columns"]["band_rms_estimated_N"][1]:.3f}')]
    (ROOT/'paper/generated/force_numbers.tex').write_text('\n'.join(tex)+'\n')
    rows=[r'\begin{tabular}{llrr}\toprule',r'Weighting & Window [Hz] & P6 error [\%] & P8 error [\%]\\\midrule']
    for c in selection['candidates']:
        label=('training error spectrum' if c['window_hz'] is not None else 'signal level')
        if c['window_hz']==window:label+=' (selected)'
        rows.append(f"{label} & {'--' if c['window_hz'] is None else f'{c[chr(119)+chr(105)+chr(110)+chr(100)+chr(111)+chr(119)+chr(95)+chr(104)+chr(122)]:g}'} & {100*c['errors'][0]:.1f} & {100*c['errors'][1]:.1f}"+r'\\')
    rows.append(r'\bottomrule\end{tabular}')
    (ROOT/'paper/generated/force_selection.tex').write_text('\n'.join(rows)+'\n')
    print(json.dumps(dict(selected_window_hz=window,selection=[(r['window_hz'],r['errors']) for r in selection['candidates']],
        results={c:{k:r[k] for k in ['record','relative_rms_error','relative_rms_error_signal_level','correlation','band_rms_measured_N','band_rms_estimated_N','fit_fraction_median','condition_quantiles','both_columns']} for c,r in reports.items()}),indent=1))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh',action='store_true',help='Re-read the external archive and overwrite the cache.')
    main(parser.parse_args().refresh)
