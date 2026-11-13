"""Shared figure legends in reserved space, separate from plotted data."""
import math


def shared_legend(fig, *, fontsize=8.5, ncol=None):
    entries={}
    for ax in fig.axes:
        handles,labels=ax.get_legend_handles_labels()
        for handle,label in zip(handles,labels):
            if label and not label.startswith('_'):entries.setdefault(label,handle)
        if ax.get_legend() is not None:ax.get_legend().remove()
    for legend in list(fig.legends):legend.remove()
    if not entries:return
    if ncol is None:ncol=min(3,len(entries)) if len(entries)!=4 else 2
    rows=math.ceil(len(entries)/ncol)
    reserved=(.18+.18*rows)/fig.get_figheight()
    engine=fig.get_layout_engine()
    if engine is not None:engine.set(rect=(0,0,1,1-reserved))
    else:fig.subplots_adjust(top=1-reserved)
    fig.legend(list(entries.values()),list(entries),loc='upper center',
        bbox_to_anchor=(.5,1),ncol=ncol,fontsize=fontsize,frameon=False,
        handlelength=2.4,columnspacing=1.4,borderaxespad=.2)


import numpy as np


def phase_lines(frequency, values):
    """Insert NaN separators at wraps, keeping every original finite sample."""
    phase = np.angle(values, deg=True)
    jumps = np.flatnonzero(abs(np.diff(phase)) > 180) + 1
    x = np.insert(np.asarray(frequency), jumps,
                  (frequency[jumps-1]+frequency[jumps])/2)
    y = np.insert(phase, jumps, np.nan)
    return x, y

