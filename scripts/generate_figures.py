"""Shared publication figures and tables from frozen numerical arrays.

Replays the primary fit and the separately selected holdout refit (spatial_fullband.pdf).
Uses saved selections and numerical arrays without smoothing or sample decimation.
"""
from pathlib import Path
import argparse
import json
import math
import subprocess
import sys

import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fetune.io import savefig_atomic,json_write,digest
from figure_layout import shared_legend, phase_lines

FIG=ROOT/'figures/paper'
GEN=ROOT/'paper/generated'
OUT=ROOT/'artifacts/forced_completion'
OLD=ROOT/'artifacts/fe_primary_revision'  # undamped J_0 completion, kept for the comparison
BUILD=ROOT/'build/paper/figure_work'
LAB=ROOT/'artifacts/model/laboratory_expanded'
SELECTED=ROOT/'artifacts/robust_revision'
BLUE,GREEN,GRAY,ORANGE,PURPLE='#0072B2','#00805c','.52','#B34700','#8D4A9E'
# Print sizes for full-width figures: ticks 9.5 pt, so log-axis exponents print near 6.3 pt.
TICK,LABEL,TITLE,LEGEND=9.5,10,10,10
ASSETS=[]
PLOT_CHECKS=[]
SOURCES=set()


def data(path):
    path=ROOT/path if not isinstance(path,Path) else path
    SOURCES.add(path)
    return np.load(path)


def record(path):
    path=ROOT/path if not isinstance(path,Path) else path
    SOURCES.add(path)
    return json.loads(path.read_text())


def style():
    plt.rcParams.update({'text.usetex':True,'font.family':'serif','font.size':10,
        'axes.titlesize':10,'axes.labelsize':10,'xtick.labelsize':9.5,
        'ytick.labelsize':9.5,'legend.fontsize':9.5,
        'text.latex.preamble':r'\usepackage{amsmath,amssymb}\usepackage{lmodern}',
        'path.simplify':False,'savefig.dpi':300,
        'axes.spines.top':False,'axes.spines.right':False,
        'axes.grid':True,'grid.alpha':.18,'savefig.bbox':'tight',
        'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6})


def save(fig,path):
    path=FIG/path if isinstance(path,str) else path
    savefig_atomic(fig,path)
    plt.close(fig)
    if path.parent==FIG:ASSETS.append(path.name)


def phase(ax,frequency,values,**kwargs):
    x,y=phase_lines(frequency,values)
    ax.plot(x,y,**kwargs)
    finite=np.isfinite(y)
    assert np.array_equal(x[finite],frequency)
    assert np.array_equal(y[finite],np.angle(values,deg=True))


def magnitude(ax,frequency,values,**kwargs):
    line,=ax.semilogy(frequency,abs(values),**kwargs)
    assert np.array_equal(line.get_xdata(),frequency)
    assert np.array_equal(line.get_ydata(),abs(values))


def spring():
    schematic=ROOT/'paper/figures/three_mass.tex'
    SOURCES.add(schematic)
    subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',
        f'-output-directory={BUILD}',str(schematic)],stdout=subprocess.DEVNULL,check=True,cwd=ROOT)
    nominal=data(OUT/'spring_nominal.npz')
    generating=record('configs/three_mass.json')['parameters']
    fig,axes=plt.subplots(1,3,figsize=(6.7,2.25),layout='constrained')
    for ax,key,truth,title in zip(axes,['mass','stiffness','damping'],
        [generating['mu_kg'],generating['k_N_per_m'],generating['c_N_s_per_m']],
        ['Mass [kg]',r'Stiffness [kN/m]',r'Damping [N\,s/m]']):
        divisor=1000 if key=='stiffness' else 1
        values=nominal[key]/divisor;ids=np.arange(1,len(values)+1)
        ax.axhline(truth/divisor,color=GREEN,ls=':',lw=1.3,label='Generating value')
        ax.plot(ids,values,'o-',color=ORANGE,ms=4,lw=.9,label='Declared nominal')
        ax.set(title=title,xticks=ids,xlim=(.7,len(values)+.3),
            xlabel='Mass index' if key=='mass' else 'Connection index',
            ylim=(.91*truth/divisor,1.09*truth/divisor))
    shared_legend(fig,ncol=2,fontsize=9.5)
    save(fig,BUILD/'spring_parameters.pdf')
    spring_completion(BUILD/'spring_completion.pdf')
    # Each vector asset is tightly cropped before this fixed-width composition.
    overview=BUILD/'spring_overview.tex'
    overview.write_text(r'''\documentclass[border=0pt]{standalone}
\usepackage[T1]{fontenc}\usepackage{lmodern,graphicx}
\begin{document}\begin{minipage}{17cm}
\textbf{(a) Grounded chain and observed ports}\par\vspace{3mm}
\includegraphics[width=\linewidth]{three_mass.pdf}\par\vspace{6mm}
\textbf{(b) One declared nominal perturbation}\par\vspace{2mm}
\includegraphics[width=\linewidth]{spring_parameters.pdf}\par\vspace{7mm}
\includegraphics[width=\linewidth]{spring_completion.pdf}
\end{minipage}\end{document}
''')
    subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',str(overview)],
        cwd=BUILD,stdout=subprocess.DEVNULL,check=True)
    import shutil
    shutil.copyfile(BUILD/'spring_overview.pdf',FIG/'spring_overview.pdf');ASSETS.append('spring_overview.pdf')
    prediction=data(OUT/'spring_predictions.npz');frequency=prediction['frequency']
    fig=plt.figure(figsize=(6.7,4.35),layout='constrained')
    grid=fig.add_gridspec(1,2,wspace=.09)
    for row in range(2):
        inner=grid[row].subgridspec(2,1,height_ratios=[1.45,1],hspace=.045)
        am=fig.add_subplot(inner[0]);ap=fig.add_subplot(inner[1],sharex=am)
        curves=[]
        if row==0:curves.append((prediction['training'][:,0,0],GRAY,'-','Measured endpoint'))
        curves.extend([(prediction['nominal'][:,row,0],ORANGE,'-.','Nominal mechanics'),
            (prediction['truth'][:,row,0],GREEN,':','Generating system'),
            (prediction['identified'][:,row,0],BLUE,'--','Identified + completed')])
        for values,color,linestyle,label in curves:
            magnitude(am,frequency,values,color=color,ls=linestyle,lw=.9,label=label)
            phase(ap,frequency,values,color=color,ls=linestyle,lw=.85)
        am.set_title(r'\textbf{(a)} Measured: $a_3/f_1$' if row==0 else r'\textbf{(b)} Unmeasured: $a_2/f_1$',fontsize=TITLE)
        am.set_ylabel(r'Magnitude [m\,s$^{-2}$/N]',fontsize=LABEL);am.tick_params(labelbottom=False)
        ap.set(ylabel='Phase [deg]',ylim=(-190,190),yticks=[-180,0,180],xlabel='Frequency [Hz]')
        for ax in [am,ap]:
            ax.set(xlim=(.1,12),xticks=[0,3,6,9,12]);ax.tick_params(labelsize=TICK)
            ax.xaxis.label.set_size(LABEL);ax.yaxis.label.set_size(LABEL)
    shared_legend(fig,ncol=2,fontsize=LEGEND)
    save(fig,'spring_frf.pdf')
    PLOT_CHECKS.append(dict(figure='spring_frf.pdf',samples=len(frequency),
        response_order=[3,2],nominal_source='artifacts/fe_primary_revision/spring_nominal.npz',
        predictions='artifacts/fe_primary_revision/spring_predictions.npz',
        exact_magnitude=True,exact_finite_phase=True,unmeasured_data_role='assessment only'))


def registration():
    import mesh_figure as rig
    style()
    mesh=data(ROOT/'artifacts/preserved/geometry/mesh_geometry.npz')
    nodes,quads=mesh['nodes'],mesh['quads']
    basis,camera=rig._camera(20,-122)
    flat=lambda points:np.asarray(points)@basis.T
    registered=json.loads((ROOT/'artifacts/model/laboratory_expanded/registration.json').read_text())
    ports=[p for p in registered['ports'] if p['kind']=='acceleration']
    fig=plt.figure(figsize=(6.7,8.1),layout='constrained')
    grid=fig.add_gridspec(3,2,height_ratios=[4.4,2.65,.85],hspace=.09,wspace=.035)
    ax=fig.add_subplot(grid[0,:])
    rig._surface(ax,basis,camera,nodes,quads,'#cbd4de')
    positions={39:(.23,-.64),33:(1.58,-.64),34:(.97,-.64),35:(.38,.99),
               36:(.96,.99),37:(2.06,.38),38:(2.06,.73),40:(1.52,.99),8:(-.42,.05)}
    callouts={}
    for port in ports:
        channel=port['channel']
        point=flat([port['position_m']])[0]
        fitted=port['status']=='calibration'
        color=BLUE if fitted else ORANGE
        ax.plot(*point,'o',ms=3.7,mec=color,mfc=color if fitted else 'white',
                mew=.9,zorder=9)
        if channel not in positions:
            continue
        label=f'A{channel-32}' if channel>=33 else 'ACC8'
        loc=positions[channel]
        annotation=ax.annotate(label,point,xytext=loc,color='#111111' if fitted else ORANGE,
            fontsize=9.5,ha='center',va='center',zorder=12,
            bbox=dict(boxstyle='round,pad=.13',fc='white',ec='none',alpha=.96),
            arrowprops=dict(arrowstyle='-',color=color,lw=.75,shrinkA=2,shrinkB=3))
        callouts[label]=dict(point=point.tolist(),text_position=list(loc))
    for port in [p for p in registered['ports'] if p['kind']=='force']:
        point=flat([port['position_m']])[0]
        label=port['id']
        ax.plot(*point,'^',ms=6,mec='white',mfc=GREEN,mew=.6,zorder=10)
        loc=(-.25,-.39) if label=='P6' else (2.06,-.10)
        ax.annotate(label,point,xytext=loc,color=GREEN,fontsize=9.5,ha='center',
            va='center',zorder=12,bbox=dict(boxstyle='round,pad=.12',fc='white',ec='none'),
            arrowprops=dict(arrowstyle='-',color=GREEN,lw=.8,shrinkA=2,shrinkB=4))
    # The whole projected triad sits beyond the mesh at lower left.
    origin=np.array([-.63,-.70])
    triad=[]
    for direction,label in zip(np.eye(3)*.23,['x','y','z']):
        delta=direction@basis.T
        head=origin+delta
        ax.annotate('',head,origin,arrowprops=dict(arrowstyle='-|>',color='.1',
                    lw=.85,shrinkA=0,shrinkB=0),zorder=11)
        label_point=origin+1.28*delta
        ax.text(*label_point,'$'+label+'$',fontsize=9.5,ha='center',va='center')
        triad.append(dict(axis=label,tail=origin.tolist(),head=head.tolist(),
                          label=label_point.tolist()))
    ax.set(xlim=(-.9,2.31),ylim=(-.91,1.14),aspect='equal')
    ax.set_axis_off()
    ax.set_anchor('N')
    ax.set_title(r'\textbf{(a)} FE mesh and registered ports',loc='left',pad=8)
    for marker,face,color,label in [
        ('o',BLUE,BLUE,'31 fitted responses'),('o','white',ORANGE,'2 excluded responses'),
        ('^',GREEN,GREEN,'2 force inputs')]:
        ax.plot([],[],marker=marker,ls='',mfc=face,mec=color,ms=5,label=label)
    photo_crops={}
    # Equal-aspect crops retain the established instrumented surfaces; no labels
    # are assigned to hardware not identified in the preserved metadata.
    for col,name,title,crop in [(0,'IMG_2709.jpg',r'\textbf{(b)} Instrumented shell',(0,90,1039,832)),
                              (1,'IMG_2710.jpg',r'\textbf{(c)} Machinery seats',(0,45,1256,942))]:
        path=ROOT/'artifacts/preserved/photos'/name;SOURCES.add(path)
        photo=fig.add_subplot(grid[1,col]);photo.imshow(Image.open(path).crop(crop))
        photo.set(xticks=[],yticks=[],title=title)
        for spine in photo.spines.values():spine.set_visible(False)
        photo_crops[name]=list(crop)
    flow=fig.add_subplot(grid[2,:]);flow.set(xlim=(0,1),ylim=(0,1));flow.axis('off')
    labels=['Two shakers','Measured forces\nP6, P8','Structure','Accelerations']
    centers=[.07,.35,.64,.91]
    for center,label in zip(centers,labels):
        flow.text(center,.55,label,ha='center',va='center',fontsize=9.5,
            bbox=dict(boxstyle='round,pad=.35',facecolor='white',edgecolor='.65',linewidth=.65))
    for left,right in zip(centers[:-1],centers[1:]):
        flow.annotate('',(right-.09,.55),(left+.09,.55),arrowprops=dict(arrowstyle='->',color='.35',lw=.8))
    flow.text(.5,.03,r'Frame: deck $+z$; nominal shell normals; A-series $-y$; forces $+z$',
        ha='center',va='center',fontsize=9)
    shared_legend(fig,fontsize=9.5,ncol=3)
    fig.canvas.draw()
    mesh_bounds=np.array([flat(nodes).min(0),flat(nodes).max(0)])
    # Audit A1/A7 text boxes in data coordinates, not just their anchor points.
    for label in ['A1','A7']:
        annotation=next(a for a in ax.texts if a.get_text()==label)
        bounds=annotation.get_bbox_patch().get_window_extent(fig.canvas.get_renderer())
        data_bounds=ax.transData.inverted().transform(bounds.get_points())
        assert data_bounds[1,1]<mesh_bounds[0,1],(label,data_bounds,mesh_bounds)
        callouts[label]['label_bbox_data']=data_bounds.tolist()
    save(fig,FIG/'experimental_overview.pdf')
    return dict(callouts=callouts,triad=triad,mesh_projected_bounds=mesh_bounds.tolist(),
                photograph_crops=photo_crops,
                note='Unified overview; external A1/A7 callouts and triad; photography remains inverted relative to the deck +z registration frame; no registration change.')


def pair_grid(cases,name,frequency,measured,curves,channels,*,height,fullband=False,shading=None,compact=False):
    """One comparison per cell, each with magnitude above principal phase."""
    rows=math.ceil(len(cases)/2)
    fig=plt.figure(figsize=(6.7,height),layout='constrained')
    grid=fig.add_gridspec(rows,2,hspace=.09,wspace=.055)
    checks=[]
    for slot,case in enumerate(cases):
        row,col=divmod(slot,2);channel=case['channel'];force=case['input_column']
        j=list(channels).index(channel)
        take=slice(None) if fullband else slice(case['sample_indices_inclusive'][0],case['sample_indices_inclusive'][1]+1)
        ff=frequency[take]
        inner=grid[row,col].subgridspec(2,1,height_ratios=[1.45,1],hspace=.045)
        am=fig.add_subplot(inner[0]);ap=fig.add_subplot(inner[1],sharex=am)
        for values,color,ls,label in [(measured,GRAY,'-','Measured, record 103'),*curves]:
            magnitude(am,ff,values[take,j,force],color=color,ls=ls,lw=.65 if fullband else .9,label=label)
            phase(ap,ff,values[take,j,force],color=color,ls=ls,lw=.58 if fullband else .85)
        title=rf'ACC{channel}$\leftarrow$P{6+2*force}'
        if not fullband or compact:title=case['label']+': '+title
        if 'title' in case:title=case['title']
        am.set_title(title,pad=3,fontsize=TITLE if compact else 10)
        am.tick_params(labelbottom=False)
        am.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        am.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10,numticks=3))
        am.set_xlim(ff[0],ff[-1]);ap.set(ylim=(-190,190),yticks=[-180,0,180])
        if fullband:
            ap.set_xticks([400,600,800,1000])
            if row<rows-1:ap.tick_params(labelbottom=False)
        if shading:
            for detail in shading:
                if detail['channel']==channel and detail['input_column']==force:
                    lo,hi=detail['frequency_hz']
                    for ax in [am,ap]:ax.axvspan(lo,hi,color=ORANGE,alpha=.10,lw=0)
        if row==rows-1:ap.set_xlabel('Frequency [Hz]')
        if col==0:
            # Short axes in the close-up grid need the compact magnitude label.
            am.set_ylabel(r'$|H|$ [m\,s$^{-2}$/N]' if compact and not fullband else r'Magnitude [m\,s$^{-2}$/N]')
            ap.set_ylabel('Phase [deg]')
        if compact:
            if not fullband:
                ap.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4,steps=[1,2,5,10]))
            for ax in [am,ap]:
                ax.tick_params(labelsize=TICK)
                ax.xaxis.label.set_size(LABEL);ax.yaxis.label.set_size(LABEL)
        checks.append(dict(channel=int(channel),input_column=force,samples=len(ff),
            frequency_hz=[float(ff[0]),float(ff[-1])],exact_magnitude=True,exact_finite_phase=True))
    shared_legend(fig,fontsize=LEGEND,ncol=min(3,1+len(curves)))
    save(fig,name)
    PLOT_CHECKS.append(dict(figure=name,cases=checks,phase_rule='NaN separators at wraps; every finite sample unchanged'))


def main_fullband(cases,frequency,measured,prediction,channels,name='frf_fullband.pdf'):
    fig=plt.figure(figsize=(6.7,8.55),layout='constrained')
    grid=fig.add_gridspec(2,1,hspace=.10)
    for row,case in enumerate(cases):
        j=list(channels).index(case['channel']);i=case['input_column']
        inner=grid[row].subgridspec(2,1,height_ratios=[1.45,1],hspace=.045)
        am=fig.add_subplot(inner[0]);ap=fig.add_subplot(inner[1],sharex=am)
        for values,color,ls,label in [(measured,GRAY,'-','Measured, record 103'),(prediction,BLUE,'--','Selected model')]:
            magnitude(am,frequency,values[:,j,i],color=color,ls=ls,lw=.8,label=label)
            phase(ap,frequency,values[:,j,i],color=color,ls=ls,lw=.7)
        am.set_title(rf'\textbf{{({chr(97+row)})}} ACC{case["channel"]}$\leftarrow$P{6+2*i}')
        am.set_ylabel(r'Magnitude [m\,s$^{-2}$/N]');am.tick_params(labelbottom=False)
        am.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ap.set(ylabel='Phase [deg]',ylim=(-190,190),yticks=[-180,0,180],
               xlim=(360,1000),xticks=[400,500,600,700,800,900,1000],xlabel='Frequency [Hz]')
        for ax in [am,ap]:ax.axvspan(*case['frequency_hz'],color=ORANGE,alpha=.10,lw=0)
    shared_legend(fig,fontsize=10)
    save(fig,name)
    PLOT_CHECKS.append(dict(figure=name,cases=cases,samples=len(frequency),
                           exact_magnitude=True,exact_finite_phase=True))

def spring_completion(path):
    z=data(OUT/'spring_illustration.npz')
    fig=plt.figure(figsize=(6.7,3.8),layout='constrained')
    grid=fig.add_gridspec(1,2,width_ratios=[1.15,1],wspace=.12)
    ax=fig.add_subplot(grid[0]);svax=fig.add_subplot(grid[1])
    ax.set(xlim=(-.4,2.55),ylim=(-.45,2.85));ax.axis('off')
    ax.set_title(r'\textbf{(c)} Restore the missing field',loc='left',pad=12)
    for baseline,values,color in [(2.0,z['before'],ORANGE),(.35,z['after'],BLUE)]:
        xpos=np.arange(3);shift=.68*values
        ax.plot(xpos,np.full(3,baseline),color='.7',ls=':',lw=.8)
        ax.plot(xpos,baseline+shift,color=color,lw=1.1,zorder=3)
        ax.plot([0,2],[baseline,baseline],'o',mfc=BLUE,mec='white',mew=.6,ms=6,zorder=5)
        ax.plot(1,baseline,'o',mfc='white',mec='.35',ms=5,zorder=4)
        if baseline<1:
            ax.annotate('',(1,baseline+shift[1]),(1,baseline),
                arrowprops=dict(arrowstyle='-|>',lw=1.4,color=BLUE),zorder=6)
            ax.text(1.12,baseline+shift[1]+.10,rf'$({values[1]:.4f})\,e_2$',color=BLUE,ha='center',va='bottom',fontsize=10)
        for i in [0,2]:ax.text(i,baseline-.21,str(i+1),ha='center',va='top',fontsize=9.5)
    ax.text(0,2.48,r'$EWv=0$',color=ORANGE,fontsize=11)
    ax.text(0,1.24,r'$T_\theta v\ne0$',color=BLUE,fontsize=11)
    ax.annotate('',(2.3,.8),(2.3,1.75),arrowprops=dict(arrowstyle='->',color='.3',lw=1.0))
    ax.text(1,-.22,r'$AT_\theta=AEW=W$',ha='center',va='top',fontsize=10.5)
    ax.text(1,2.78,r'$v\in\ker W$ is a latent direction; dots are physical nodes',ha='center',fontsize=9)
    svax.plot([1,2,3],z['singular_before'],'o--',color=ORANGE,ms=5,label=r'Static lift $EW$')
    svax.plot([1,2,3],z['singular_after'],'s-',color=BLUE,ms=4.7,label=r'Completed $T_\theta$')
    floor=record(OUT/'spring_construction.json')['required_sigma_min']
    svax.axhline(floor,color=GREEN,ls=':',lw=.9,label=r'Required $\delta$')
    svax.set(title=r'\textbf{(d)} Verify independence',xlabel='Singular-value index',
        ylabel=r'Singular value [kg$^{-1/2}$]',xticks=[1,2,3],ylim=(-.07,1.44),yticks=[0,.5,1.0])
    svax.text(2.45,.08,'Rank 2',color=ORANGE,ha='right',fontsize=10)
    svax.text(2.9,1.13,'Rank 3',color=BLUE,ha='right',fontsize=10)
    svax.text(1.18,.18,rf'$\delta={floor:.6f}$',color=GREEN,fontsize=9)
    svax.text(2.85,.81,rf'$\sigma_3={z["singular_after"][-1]:.5f}$',color=BLUE,ha='right',fontsize=9)
    shared_legend(fig,fontsize=9.5,ncol=3)
    save(fig,path)
    PLOT_CHECKS.append(dict(figure='spring_overview.pdf',source='artifacts/fe_primary_revision/spring_illustration.npz',
        latent_direction=z['v0'].tolist(),before=z['before'].tolist(),after=z['after'].tolist(),
        singular_before=z['singular_before'].tolist(),singular_after=z['singular_after'].tolist()))


def laboratory(completion_only=False):
    observed=data(LAB/'observations.npz');saved=data(OUT/'laboratory_predictions.npz')
    frequency=saved['frequency'];prediction=saved['prediction'];measured=observed['test_103'];channels=observed['channels']
    display=record(LAB/'display_selection.json')
    cases=[next(c for c in display['fit_cases'] if c['input_column']==0),
           next(c for c in reversed(display['fit_cases']) if c['input_column']==1)]
    assert [c['label'] for c in cases]==['A','G']
    by_label={case['label']:case for case in display['fit_cases']}
    expanded_groups=[('frf_fullband_bc.pdf',['B','C']),('frf_fullband_df.pdf',['D','F'])]
    full_cases=list(cases)
    for name,labels in expanded_groups:
        group=[by_label[label] for label in labels];full_cases.extend(group)
        if not completion_only:main_fullband(group,frequency,measured,prediction,channels,name)
    assert sorted(case['channel'] for case in full_cases)==display['representative_channels']
    if not completion_only:
        pair_grid(sorted(full_cases,key=lambda case:case['label']),'frf_fullband.pdf',frequency,measured,
            [(prediction,BLUE,'--','Selected model')],channels,height=9.55,
            fullband=True,shading=display['fit_cases'],compact=True)
    for name,group in [('frf_details_ad.pdf',display['fit_cases']),
                       ('frf_details_eh.pdf',display['fit_cases'][4:])]:
        if not completion_only:
            pair_grid(group,name,frequency,measured,
                [(prediction,BLUE,'--','Selected model')],channels,
                height=9.6 if len(group)==8 else 8.55,compact=len(group)==8)
    spatial_plot=spatial_holdouts()
    held_full=spatial_plot['cases']
    spatial_assessment(observed,saved)
    return dict(primary_fit='artifacts/robust_revision/final_polish.npz',
        completion='artifacts/fe_primary_revision/laboratory_completion.npz',
        main_cases=full_cases,detail_cases=display['fit_cases'],spatial_cases=held_full,
        spatial_display=spatial_plot['scope'],spatial_completion=spatial_plot['completion'],
        full_samples=len(frequency),phase_preservation='principal values retained; NaN wrap gaps',
        numerical_evidence='artifacts/model/laboratory_expanded; artifacts/fe_primary_revision')


def spatial_holdouts():
    """Holdout figure: the separately refitted spatial holdout pair, drawn like Figure 5."""
    config=record(ROOT/'configs/figure8.json');study=ROOT/config['study']
    selection=record(study/'selection.json');dest=study/config['trial']
    assert config['trial']==selection['selected_trial']
    assert config['channels']==selection['selected_channels']
    assert config['selection_record']==selection['screen_record']
    assert config['record']=='test_103'  # pair_grid labels the measurement as record 103
    assessment=record(dest/'assessment.json');fit=data(dest/'fit.npz')
    assert digest(dest/'completion.npz')==assessment['completion_sha256']
    assert digest(dest/'fit.npz')==assessment['fit_sha256']
    SOURCES.add(dest/'completion.npz')
    observed=data(LAB/'observations.npz');saved=data(dest/'predictions.npz')
    frequency=saved['frequency'];channels=saved['channels'];prediction=saved['prediction']
    measured=observed[config['record']];displayed=config['channels']
    assert np.array_equal(channels,observed['channels'])
    assert np.array_equal(frequency,observed['frequency'])
    assert set(displayed)<=set(channels[fit['held_rows']])
    assert not set(displayed)&set(channels[fit['fit_rows']])
    # Rows are the inputs P6, P8; columns the withheld channels. Each cell pairs
    # magnitude and principal phase, exactly as in frf_fullband.pdf.
    cases=[]
    for force in [0,1]:
        for col,channel in enumerate(displayed):
            letter=chr(97+2*force+col)
            cases.append(dict(channel=channel,input_column=force,label=letter,
                title=rf'\textbf{{({letter})}} ACC{channel}$\leftarrow$P{6+2*force}'))
    pair_grid(cases,'spatial_fullband.pdf',frequency,measured,
        [(prediction,BLUE,'--','FE completion, withheld sensors')],channels,
        height=6.6,fullband=True,compact=True)
    metrics=[]
    for channel in displayed:
        reference=next(v for v in assessment['rows'] if v['channel']==channel and v['record']==config['record'])
        metrics.append(dict(channel=channel,complex_error=reference['complex_error']))
    check=PLOT_CHECKS[-1]
    assert check['figure']=='spatial_fullband.pdf'
    check.update(prediction_source=str((dest/'predictions.npz').relative_to(ROOT)),
        completion=str((dest/'completion.npz').relative_to(ROOT)),
        trial=config['trial'],scope=config['role'],selection_record=config['selection_record'],
        assessment_record=config['record'],held_channels=channels[fit['held_rows']].tolist(),
        fitted_channels=channels[fit['fit_rows']].tolist(),metrics=metrics,
        layout='as frf_fullband.pdf: magnitude above principal phase in every cell',
        columns=[f'ACC{c}' for c in displayed],rows=['P6','P8'])
    tex=[r'\newcommand{\FigureEight'+side+'Error}{'+f'{100*m["complex_error"]:.2f}'+'}'
         for side,m in zip(['Left','Right'],metrics)]
    (GEN/'figure8_numbers.tex').write_text('\n'.join(tex)+'\n')
    json_write(ROOT/'artifacts/figure8_revision/plot_checks.json',check)
    return check


def spatial_assessment(observed,saved):
    """Reporting-only assessment of the frozen primary prediction at excluded ports."""
    held=observed['held_rows'];channels=observed['channels'];prediction=saved['prediction']
    measured=observed['test_103'];scale=saved['scale'];frequency=saved['frequency']
    rows=[]
    for j in held:
        for force in range(2):
            difference=prediction[:,j,force]-measured[:,j,force]
            error=float(np.sqrt(np.mean(abs(difference/scale[j,force])**2)))
            rms=float(np.sqrt(np.mean(abs(difference)**2)))
            phase_error=float(np.median(abs(np.angle(prediction[:,j,force]*measured[:,j,force].conj(),deg=True))))
            rows.append(dict(channel=int(channels[j]),force=f'P{6+2*force}',input_column=force,
                normalized_complex_error=error,absolute_rms=rms,median_absolute_phase_deg=phase_error,
                reporting_scale=float(scale[j,force]),samples=len(frequency)))
    aggregate=float(np.sqrt(np.mean([row['normalized_complex_error']**2 for row in rows])))
    old=record(OUT/'laboratory_assessment.json')['excluded']
    assert np.isclose(aggregate,old['complex'],rtol=1e-13)
    assert np.allclose(np.array([row['normalized_complex_error'] for row in rows]).reshape(2,2),old['per_channel'],rtol=1e-13)
    json_write(OUT/'spatial_display_assessment.json',dict(record='test_103',
        model='artifacts/fe_primary_revision/laboratory_completion.npz',
        aggregate_complex_error=aggregate,transfers=rows,display='full band only',
        scope='Reporting only. Same accepted FE completion; no coefficient, prior, scale or window tuning.'))
    lines=[r'\begin{tabular}{lrrr}\toprule',
        r'Transfer & Complex error [\%] & RMS error [m\,s$^{-2}$/N] & Median phase error [$^\circ$]\\\midrule']
    for row in rows:
        lines.append(f'ACC{row["channel"]}/{row["force"]} & {100*row["normalized_complex_error"]:.2f} & {row["absolute_rms"]:.3f} & {row["median_absolute_phase_deg"]:.1f}'+r'\\')
    lines.append(r'\bottomrule\end{tabular}')
    (GEN/'primary_spatial_results.tex').write_text('\n'.join(lines)+'\n')




def force_estimation():
    """Appendix figures: forces at known positions estimated from accelerations."""
    two=data(ROOT/'artifacts/force_estimation/two_force.npz');one=data(ROOT/'artifacts/force_estimation/one_force.npz')
    meta_two=record(ROOT/'artifacts/force_estimation/two_record.json');meta_one=record(ROOT/'artifacts/force_estimation/one_record.json')
    record(ROOT/'artifacts/force_estimation/selection.json')
    def panels(fig,grid,row,arrays,column,label):
        t=(arrays['time']-arrays['time'][0])*1e3
        at=fig.add_subplot(grid[row,0]);ap=fig.add_subplot(grid[row,1])
        measured=arrays['measured_display'][:,column];estimated=arrays['estimated_display'][:,column]
        line,=at.plot(t,measured,color=GRAY,lw=.9,label='Measured force');assert np.array_equal(line.get_ydata(),measured)
        line,=at.plot(t,estimated,color=BLUE,ls='--',lw=.9,label=r'Estimated $f_\theta$');assert np.array_equal(line.get_ydata(),estimated)
        at.set(xlim=(0,t[-1]),xlabel='Time [ms]',ylabel='Force [N]',title=label+', 50 ms window')
        # Display spectra: Welch averages over the assessment interval (6--58 s)
        # with 1 s Hann windows and 50% overlap, from the saved band-limited
        # waveforms; ~100 averages give readable level comparisons at 1 Hz.
        from scipy.signal import welch
        rate=8192.;keep=slice(int(6*rate),int(58*rate)+1)
        def display_psd(x):
            fw,pw=welch(np.asarray(x[keep,column],float),fs=rate,window='hann',nperseg=8192,
                noverlap=4096,detrend='constant',scaling='density')
            # Interior lines only: the two 1 Hz bins at the DFT-mask edges are partial.
            band=(fw>=362)&(fw<=998);return fw[band],pw[band]
        fm,pm=display_psd(arrays['measured']);fe,pe=display_psd(arrays['estimated'])
        assert np.array_equal(fm,fe) and np.all(pm>0) and np.all(pe>0)
        line,=ap.semilogy(fm,pm,color=GRAY,lw=1.0);assert np.array_equal(line.get_ydata(),pm)
        line,=ap.semilogy(fe,pe,color=BLUE,ls='--',lw=1.0);assert np.array_equal(line.get_ydata(),pe)
        assert 1e-3<min(pm.min(),pe.min()) and max(pm.max(),pe.max())<1e-1
        ap.set_ylim(1e-3,1e-1)
        ap.set(xlim=(360,1000),xticks=[400,600,800,1000],xlabel='Frequency [Hz]',ylabel=r'Force PSD [N$^2$/Hz]',title=label.split('} ',1)[1]+', band spectrum')
        ap.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        return dict(samples=len(t),window_ms=float(t[-1]),spectral_lines=len(fm),exact_waveform=True,
            display_psd='Welch, 1 s Hann windows, 50% overlap, 6-58 s, from saved band-limited waveforms')
    fig=plt.figure(figsize=(6.7,6.0),layout='constrained');grid=fig.add_gridspec(2,2,hspace=.12,wspace=.08)
    checks=[panels(fig,grid,0,two,0,r'\textbf{(a)} P6'),panels(fig,grid,1,two,1,r'\textbf{(b)} P8')]
    shared_legend(fig,fontsize=10,ncol=2);save(fig,'force_two.pdf')
    PLOT_CHECKS.append(dict(figure='force_two.pdf',record=meta_two['record'],cases=checks,
        relative_rms_error=meta_two['relative_rms_error'],error_window_hz=meta_two['error_window_hz'],
        source='artifacts/force_estimation/two_force.npz'))
    fig=plt.figure(figsize=(6.7,3.0),layout='constrained');grid=fig.add_gridspec(1,2,wspace=.08)
    checks=[panels(fig,grid,0,one,0,r'\textbf{(a)} P6')]
    shared_legend(fig,fontsize=10,ncol=2);save(fig,'force_one.pdf')
    PLOT_CHECKS.append(dict(figure='force_one.pdf',record=meta_one['record'],cases=checks,
        relative_rms_error=meta_one['relative_rms_error'],both_columns_control=meta_one['both_columns'],
        error_window_hz=meta_one['error_window_hz'],source='artifacts/force_estimation/one_force.npz'))


def scientific(value):
    if value==0:return '0'
    a,b=f'{value:.2e}'.split('e')
    return rf'\ensuremath{{{float(a):.2f}\times10^{{{int(b)}}}}}'


def tables():
    spring_worked()
    lc=record(OUT/'laboratory_construction.json');la=record(OUT/'laboratory_assessment.json')
    sc=record(OUT/'spring_construction.json');sa=record(OUT/'spring_assessment.json')
    macros=dict(SpringCondition=f'{sc["rank_and_condition"]["condition"]:.4f}',
        SpringSigmaMin=f'{sc["rank_and_condition"]["minimum"]:.5f}',SpringMargin=f'{sc["required_sigma_min"]:.6f}',
        SpringHiddenFirst=f'{sa["hidden_coefficients"][0]:.5f}',SpringHiddenSecond=f'{sa["hidden_coefficients"][1]:.5f}',SpringHiddenThird=f'{sa["hidden_coefficients"][2]:.5f}',
        SpringPhysicalResidual=scientific(sa['physical_frf']),SpringMassDefect=f'{sc["mass_gram"]:.5f}',
        SpringStiffnessDefect=f'{sc["stiffness_gram"]:.5f}',
        SpringNominalEndpoint=f'{100*sa["response_assessment"]["nominal"]["endpoint"]["complex"]:.2f}',
        SpringIdentifiedEndpoint=f'{100*sa["response_assessment"]["identified"]["endpoint"]["complex"]:.3f}',
        SpringNominalMiddle=f'{100*sa["response_assessment"]["nominal"]["middle"]["complex"]:.2f}',
        SpringIdentifiedMiddle=f'{100*sa["response_assessment"]["identified"]["middle"]["complex"]:.2f}',
        SpringNominalMiddleRMS=f'{sa["response_assessment"]["nominal"]["middle"]["absolute_rms"][0][0]:.3f}',
        SpringIdentifiedMiddleRMS=f'{sa["response_assessment"]["identified"]["middle"]["absolute_rms"][0][0]:.4f}',
        SpringTrainingError=f'{100*sa["training_error"]["complex"]:.3f}',
        SpringIndependentError=f'{100*sa["independent_noisy_error"]["complex"]:.2f}',
        SpringMassEigenMin=f'{sa["prior_mass_residual_eigenvalues"][0]:.5f}',
        SpringPhysicalMassError=f'{100*sa["physical_discrepancies"]["M"]:.2f}',
        SpringPhysicalDampingError=f'{100*sa["physical_discrepancies"]["C"]:.2f}',
        SpringPhysicalStiffnessError=f'{100*sa["physical_discrepancies"]["K"]:.2f}',
        LabSigmaMin=f'{lc["rank_and_condition"]["minimum"]:.5f}',LabCondition=f'{lc["rank_and_condition"]["condition"]:.1f}',
        LabPCondition=f'{lc["P"]["condition"]:.1f}',LabMassCondition=scientific(lc['P']['condition']**2),
        LabMargin=f'{lc["required_sigma_min"]:.5f}',LabPrimalResidual=scientific(lc['criterion']['constrained']['primal_relative']),
        LabDualResidual=scientific(lc['criterion']['constrained']['dual_relative']),
        LabAnchorResidual=scientific(lc['anchor']),LabPortResidual=scientific(la['registered_all_lines']),
        LabPhysicalResidual=scientific(la['physical_frf']),
        LabStateResidual=scientific(max(v for row in la['state_space'].values() for v in row.values())),
        LabExcludedError=f'{100*la["excluded"]["complex"]:.2f}')
    fc=lc['forced'];con=lc['criterion']['constrained'];diag=lc['physical_diagnostics'];weight=fc['modal_weight']
    Pi=data(OUT/'laboratory_completion.npz')['Pi'];w=data(OUT/'laboratory_completion.npz')['omega']
    share=w**2*np.diag(Pi);share=share/share.sum();D=np.diag(np.diag(Pi)**-.5);corr=D@Pi@D;np.fill_diagonal(corr,0)
    old=record(OLD/'laboratory_assessment.json');oc=data(OLD/'laboratory_completion.npz');nc=data(OUT/'laboratory_completion.npz')
    from fetune.fe_completion import forced_objective
    dictionary=data(OUT/'laboratory_dictionary.npz');ops=[dictionary['KP'],dictionary['MP'],dictionary['HP']]
    macros.update(LabJF=f'{fc["J_F_accepted"]:.5f}',LabJFLS=f'{fc["J_F_equality_only"]:.5f}',
        LabJFStart=f'{con["start_J_F"]:.5f}',
        LabLSMinimum=scientific(fc['equality_only_sigma_min']),LabIterations=str(con['iterations']),
        LabMarginCost=f'{100*(fc["J_F_accepted"]/fc["J_F_equality_only"]-1):.2f}',
        LabProjectionAgreement=scientific(lc['objective_reference']['full_vs_projected_LS_relative']),
        LabLeftInverse=scientific(diag['left_inverse_frobenius']),LabWorstPort=scientific(diag['port_worst_column_relative']),
        LabShareMin=scientific(share.min()),LabShareMax=f'{share.max():.4f}',
        LabCorrelationMax=f'{abs(corr).max():.2f}',LabCorrelatedPairs=str(int((abs(corr)>.5).sum()//2)),
        LabJZeroEquivalent=f'{fc["J0_equivalent_accepted"]:.3f}',
        LabJFOfJZero=f'{forced_objective(oc["W"],oc["Z"],oc["omega"],ops,Pi):.5f}',
        LabMapChange=f'{100*np.linalg.norm(nc["T"]-oc["T"])/np.linalg.norm(oc["T"]):.0f}',
        LabExcludedErrorJZero=f'{100*old["excluded"]["complex"]:.2f}',
        LabMassDefect=f'{lc["mass_gram"]:.1f}',LabMassDefectJZero=f'{record(OLD/"laboratory_construction.json")["mass_gram"]:.2f}')
    (GEN/'primary_fe_numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')
    rows=[('Training / monitoring / development error',
        ' / '.join(f'{100*la["scores"][key]["complex"]:.2f}' for key in ['test_96','test_101','test_103'])+r'\%'),
        (r'$\rank W$ / $\rank T_\theta$',f'{lc["rank_W"]} / {lc["rank_and_condition"]["rank"]}'),
        (r'Hidden fields $d$',str(lc['hidden_fields'])),
        (r'$\sigma_r(RT_\theta)$ / margin $\delta$',f'{lc["rank_and_condition"]["minimum"]:.5f} / {lc["required_sigma_min"]:.5f}'),
        (r'$\kappa_2(RT_\theta)$ / $\kappa_2(P)$',f'{lc["rank_and_condition"]["condition"]:.1f} / {lc["P"]["condition"]:.1f}'),
        (r'Physical/normalized FRF discrepancy',scientific(la['physical_frf']))]
    lines=[r'\begin{tabular}{ll}\toprule',r'Quantity & Primary result\\\midrule']
    lines.extend(label+' & '+value+r'\\' for label,value in rows);lines.append(r'\bottomrule\end{tabular}')
    (GEN/'primary_fe_results.tex').write_text('\n'.join(lines)+'\n')
    checks=[(r'Interpolation $AT_\theta=W$',lc['anchor']),('Full-band registered FRFs',la['registered_all_lines']),
        ('Physical force transformation',la['force']),('Physical response transformation',la['output']),
        ('Acceleration feedthrough',la['feedthrough']),('Maximum mechanical congruence residual',max(la['congruences'].values())),
        (r'Nominal mass Gram defect $\|T_\theta^{\top}M_0T_\theta-I\|_F/\|I\|_F$',lc['mass_gram']),
        (r'Nominal stiffness Gram defect $\|T_\theta^{\top}K_0T_\theta-\Omega^2\|_F/\|\Omega^2\|_F$',lc['stiffness_gram'])]
    lines=[r'\begin{tabular}{lr}\toprule',r'Check & Relative value\\\midrule']
    lines.extend(label+' & '+(scientific(value) if value<.01 else f'{value:.4f}')+r'\\' for label,value in checks)
    lines.append(r'\bottomrule\end{tabular}')
    (GEN/'primary_fe_verification.tex').write_text('\n'.join(lines)+'\n')


def spring_worked():
    """Readable numbers generated directly from the accepted full-precision fit."""
    z=data(OUT/'spring_completion.npz');prior=data(OUT/'spring_nominal.npz')
    def matrix(a,digits=6):
        rows=[' & '.join(f'{v:.{digits}f}' if abs(v)>=.5*10**-digits else '0'
                         for v in row) for row in np.atleast_2d(a)]
        return r'\begin{bmatrix}'+r'\\'.join(rows)+r'\end{bmatrix}'
    values={name:matrix(z[key],digits) for name,key,digits in
            [('W','W',6),('Gamma','Gamma',6),('Z','Z',6),('T','T',6),
             ('M','M',6),('C','C',5),('K','K',3)]}
    values['OmegaSquared']=', '.join(f'{v:.6f}' for v in z['omega']**2)
    Pi=z['Pi'];w=z['omega'];K,M=prior['K'],prior['M'];n=z['N_full'][:,0];EW=z['E_full']@z['W']
    c=[(K-wj**2*M)@n for wj in w];a=[(K-wj**2*M)@EW[:,j] for j,wj in enumerate(w)]
    H=np.array([[Pi[j,k]*c[j]@la.solve(K,c[k]) for k in range(3)] for j in range(3)])
    bvec=np.array([sum(Pi[j,k]*c[j]@la.solve(K,a[k]) for k in range(3)) for j in range(3)])
    assert np.allclose(-la.solve(H,bvec),z['Z'][0],rtol=1e-9,atol=1e-12)
    values['Pi']=matrix(Pi*1e6,4);values['H']=matrix(H*1e3,4);values['b']=matrix(bvec[None,:]*1e3,4)
    D=np.diag(np.diag(Pi)**-.5);corr=D@Pi@D;np.fill_diagonal(corr,0)
    values['PiCorrelation']=f'{abs(corr).max():.3f}'
    def JF(T):
        G=K@T-(M@T)*w**2
        return np.sum((G.T@la.solve(K,G))*Pi)/np.sum(w**2*np.diag(Pi))
    values['JFStatic']=f'{JF(z["T0"]):.4f}';values['JF']=scientific(JF(z['T']))
    old=data(OLD/'spring_completion.npz')
    values['MapChange']=f'{100*la.norm(z["T"]-old["T"])/la.norm(old["T"]):.2f}'
    values['OldMiddle']=f'{100*record(OLD/"spring_assessment.json")["response_assessment"]["identified"]["middle"]["complex"]:.2f}'
    construction=record(OUT/'spring_construction.json')
    values['LeftInverse']=scientific(construction['physical_diagnostics']['left_inverse_frobenius'])
    (GEN/'spring_worked.tex').write_text('\n'.join(
        r'\newcommand{\Spring'+key+'}{'+value+'}' for key,value in values.items())+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completion-only',action='store_true',help='Preserve rig, fitted-port and force figures byte-for-byte.')
    parser.add_argument('--figure8-only','--figure10-only',dest='figure8_only',action='store_true',
        help='Redraw Figure 8 from the saved selected holdout refit.')
    args=parser.parse_args()
    for directory in [FIG,GEN,OUT,BUILD]:directory.mkdir(parents=True,exist_ok=True)
    if args.completion_only:
        style();spring();display=laboratory(completion_only=True);tables()
        provenance=json.loads((OUT/'figure_provenance.json').read_text())
        provenance['plot_checks']=[v for v in provenance['plot_checks'] if v['figure'] not in ASSETS]+PLOT_CHECKS
        provenance['sources'].update({str(p.relative_to(ROOT)):digest(p) for p in sorted(SOURCES|{Path(__file__)})})
        provenance['display']=display
        provenance['scope']='Primary model for the main results; Figure 8 uses the separately refitted holdout model specified in configs/figure8.json.'
        provenance['completion_revision']='Zero-ridge injective completion; all r modal directions checked'
        json_write(OUT/'figure_provenance.json',provenance)
        json_write(ROOT/'artifacts/injective_revision/figure_manifest.json',
            {str(p.relative_to(ROOT)):digest(p) for p in sorted(FIG.glob('*.pdf'))})
        print('Regenerated completion-dependent figures:',ASSETS,flush=True)
        return
    if args.figure8_only:
        style();spatial_plot=spatial_holdouts()
        provenance=json.loads((OUT/'figure_provenance.json').read_text())
        provenance['plot_checks']=[v for v in provenance['plot_checks'] if v['figure']!='spatial_fullband.pdf']+PLOT_CHECKS
        SOURCES.add(Path(__file__))
        provenance['sources'].update({str(p.relative_to(ROOT)):digest(p) for p in sorted(SOURCES)})
        provenance['display'].update(spatial_cases=spatial_plot['cases'],
            spatial_display=spatial_plot['scope'],spatial_completion=spatial_plot['completion'])
        provenance['scope']='Primary model for the main results; Figure 8 uses the separately refitted holdout model specified in configs/figure8.json.'
        json_write(OUT/'figure_provenance.json',provenance)
        json_write(ROOT/'artifacts/injective_revision/figure_manifest.json',
            {str(p.relative_to(ROOT)):digest(p) for p in sorted(FIG.glob('*.pdf'))})
        print('Regenerated Figure 8:',spatial_plot['columns'],'from',spatial_plot['trial'],flush=True)
        return
    style();spring();overview=registration();display=laboratory();force_estimation();tables()
    SOURCES.update([Path(__file__),ROOT/'scripts/mesh_figure.py',ROOT/'scripts/figure_layout.py',
        ROOT/'artifacts/preserved/geometry/mesh_geometry.npz',LAB/'registration.json'])
    json_write(OUT/'figure_provenance.json',dict(figures=ASSETS,display=display,experimental_overview=overview,plot_checks=PLOT_CHECKS,
        sources={str(p.relative_to(ROOT)):digest(p) for p in sorted(SOURCES)},
        identity=dict(primary_color=BLUE,primary_style='dashed',measurement_color=GRAY,anchor_color=ORANGE),
        scope='Primary FE-completed model, with the selected separate holdout refit in Figure 8. Frozen reporting scales and all samples retained.'))
    json_write(ROOT/'artifacts/clarity_revision/figure_manifest.json',
        {str(p.relative_to(ROOT)):digest(p) for p in sorted(FIG.glob('*.pdf'))})
    print('Regenerated',len(ASSETS),'figure assets (including retained alternate layouts).',flush=True)


if __name__=='__main__':main()
