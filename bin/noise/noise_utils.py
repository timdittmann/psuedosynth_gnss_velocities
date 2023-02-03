import pandas as pd
from obspy import read
import obspy
from obspy.signal import PPSD
from numpy import where,mean
from glob import glob
from matplotlib import pyplot as plt

import numpy
import datetime
import calendar
import math
#import georinex as gr
import obspy
from obspy.io.sac import SACTrace
import numpy as np
import os

import numpy as np

def psd_stack2hist_stack(psd_stack_array, Eppsd):
    '''
    ADD DOC
    '''

    # determine which psd pieces should be used in the stack,
    # based on all selection criteria specified by user
    '''selected = self._stack_selection(
        starttime=starttime, endtime=endtime,
        time_of_weekday=time_of_weekday, year=year, month=month,
        isoweek=isoweek, callback=callback)
    '''
    selected = np.ones(psd_stack_array.shape[0], dtype=bool)
    used_indices = selected.nonzero()[0]
    used_count = len(used_indices)
    #used_times = np.array(self._times_processed)[used_indices]

    period_bin_centers=Eppsd.period_bin_centers
    num_period_bins = len(period_bin_centers)
    num_db_bins = len(Eppsd.db_bin_centers)

    # initial setup of 2D histogram
    hist_stack = np.zeros((num_period_bins, num_db_bins), dtype=np.uint64)
 
    # concatenate all used spectra, evaluate index of amplitude bin each
    # value belongs to
    inds = np.vstack([psd_stack_array[i] for i in used_indices])
    # for "inds" now a number of ..
    #   - 0 means below lowest bin (bin index 0)
    #   - 1 means, hit lowest bin (bin index 0)
    #   - ..
    #   - len(self.db_bin_edges) means above top bin
    # we need minus one because searchsorted returns the insertion index in
    # the array of bin edges which is the index of the corresponding bin
    # plus one
    inds = Eppsd.db_bin_edges.searchsorted(inds, side="left") - 1
    # for "inds" now a number of ..
    #   - -1 means below lowest bin (bin index 0)
    #   - 0 means, hit lowest bin (bin index 0)
    #   - ..
    #   - (len(self.db_bin_edges)-1) means above top bin
    # values that are left of first bin edge have to be moved back into the
    # binning
    inds[inds == -1] = 0
    # same goes for values right of last bin edge
    inds[inds == num_db_bins] -= 1
    # reshape such that we can iterate over the array, extracting for
    # each period bin an array of all amplitude bins we have hit
    inds = inds.reshape((used_count, num_period_bins)).T
    for i, inds_ in enumerate(inds):
        # count how often each bin has been hit for this period bin,
        # set the current 2D histogram column accordingly
        hist_stack[i, :] = np.bincount(inds_, minlength=num_db_bins)

    # calculate and set the cumulative version (i.e. going from 0 to 1 from
    # low to high psd values for every period column) of the current
    # histogram stack.
    # sum up the columns to cumulative entries
    hist_stack_cumul = hist_stack.cumsum(axis=1)
    # normalize every column with its overall number of entries
    # (can vary from the number of self.times_processed because of values
    #  outside the histogram db ranges)
    norm = hist_stack_cumul[:, -1].copy().astype(np.float64)
    # avoid zero division
    norm[norm == 0] = 1
    hist_stack_cumul = (hist_stack_cumul.T / norm).T
    
    return hist_stack, hist_stack_cumul

def _setup_period_binning(Eppsd, period_smoothing_width_octaves=1.0,
                 period_step_octaves=0.125, period_limits=None):
    """
    Set up period binning.
    """
    if period_limits is None:
        period_limits = (Eppsd.psd_periods[0], Eppsd.psd_periods[-1])
    # we step through the period range at step width controlled by
    # period_step_octaves (default 1/8 octave)
    period_step_factor = 2 ** period_step_octaves
    # the width of frequencies we average over for every bin is controlled
    # by period_smoothing_width_octaves (default one full octave)
    period_smoothing_width_factor = \
        2 ** period_smoothing_width_octaves
    # calculate left/right edge and center of first period bin
    # set first smoothing bin's left edge such that the center frequency is
    # the lower limit specified by the user (or the lowest period in the
    # psd)
    per_left = (period_limits[0] /
                (period_smoothing_width_factor ** 0.5))
    per_right = per_left * period_smoothing_width_factor
    per_center = math.sqrt(per_left * per_right)
    # build up lists
    per_octaves_left = [per_left]
    per_octaves_right = [per_right]
    per_octaves_center = [per_center]
    # do this for the whole period range and append the values to our lists
    while per_center < period_limits[1]:
        # move left edge of smoothing bin further
        per_left *= period_step_factor
        # determine right edge of smoothing bin
        per_right = per_left * period_smoothing_width_factor
        # determine center period of smoothing/binning
        per_center = math.sqrt(per_left * per_right)
        # append to lists
        per_octaves_left.append(per_left)
        per_octaves_right.append(per_right)
        per_octaves_center.append(per_center)
    per_octaves_left = np.array(per_octaves_left)
    per_octaves_right = np.array(per_octaves_right)
    per_octaves_center = np.array(per_octaves_center)
    valid = per_octaves_right > Eppsd.psd_periods[0]
    valid &= per_octaves_left < Eppsd.psd_periods[-1]
    per_octaves_left = per_octaves_left[valid]
    per_octaves_right = per_octaves_right[valid]
    per_octaves_center = per_octaves_center[valid]
    _period_binning = np.vstack([
        # left edge of smoothing (for calculating the bin value from psd
        per_octaves_left,
        # left xedge of bin (for plotting)
        per_octaves_center / (period_step_factor ** 0.5),
        # bin center (for plotting)
        per_octaves_center,
        # right xedge of bin (for plotting)
        per_octaves_center * (period_step_factor ** 0.5),
        # right edge of smoothing (for calculating the bin value from psd
        per_octaves_right])
    return _period_binning

def get_percentile(hist_stack_cumul, percentile, _db_bin_edges, period_bin_centers):
        """
        Returns periods and approximate psd values for given percentile value.

        :type percentile: int
        :param percentile: percentile for which to return approximate psd
                value. (e.g. a value of 50 is equal to the median.)
        :returns: (periods, percentile_values)
        """
        hist_cum = hist_stack_cumul
        if hist_cum is None:
            return None
        # go to percent
        percentile = percentile / 100.0
        if percentile == 0:
            # only for this special case we have to search from the other side
            # (otherwise we always get index 0 in .searchsorted())
            side = "right"
        else:
            side = "left"
        percentile_values = [col.searchsorted(percentile, side=side)
                             for col in hist_cum]
        # map to power db values
        percentile_values = _db_bin_edges[percentile_values]
        return (period_bin_centers, percentile_values)
    
    

from obspy.core.util import AttribDict

def _plot_histogram(Eppsd, hist_stack,hist_stack_cumul, db_bins,draw=False, filename=None):
    import matplotlib.pyplot as plt
    
    fsize = 15
    tsize = 18
    tdir = 'in'
    major = 5.0
    minor = 3.0
    lwidth = 0.8
    lhandle = 2.0
    plt.style.use('default')

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = fsize
    plt.rcParams['legend.fontsize'] = fsize-5
    plt.rcParams['xtick.direction'] = tdir
    plt.rcParams['ytick.direction'] = tdir
    plt.rcParams['xtick.major.size'] = major
    plt.rcParams['xtick.minor.size'] = minor
    plt.rcParams['ytick.major.size'] = 5.0
    plt.rcParams['ytick.minor.size'] = 3.0
    plt.rcParams['axes.linewidth'] = lwidth
    plt.rcParams['legend.handlelength'] = lhandle

    """
    Reuse a previously created figure returned by `plot(show=False)`
    and plot the current histogram stack (pre-computed using
    :meth:`calculate_histogram()`) into the figure. If a filename is
    provided, the figure will be saved to a local file.
    Note that many aspects of the plot are statically set during the first
    :meth:`plot()` call, so this routine can only be used to update with
    data from a new stack.
    """
    
    fig = plt.figure(figsize=(6,4))
    fig.ppsd = AttribDict()
    ax = fig.add_subplot(111)
    ax = fig.axes[0]
    xlim = ax.get_xlim()
    '''if "quadmesh" in fig.ppsd:
        fig.ppsd.pop("quadmesh").remove()

    if fig.ppsd.cumulative:
        data = hist_stack_cumul * 100.0
    else:
        # avoid divison with zero in case of empty stack
        data = (
            hist_stack * 100.0 /
            (hist_stack.sum(axis=1)[0] or 1))
    '''   
    data = (
            hist_stack * 100.0 /
            (hist_stack.sum(axis=1)[0] or 1))

    _period_binning = _setup_period_binning(Eppsd)
    period_bin_centers=Eppsd.period_bin_centers
    
    xedges = np.concatenate([_period_binning[1, 0:1],
                               _period_binning[3, :]])
    #if fig.ppsd.xaxis_frequency:
    #    xedges = 1.0 / xedges
    
    # setup db binning
    # Set up the binning for the db scale.
    num_bins = int((db_bins[1] - db_bins[0]) / db_bins[2])
    _db_bin_edges = np.linspace(db_bins[0], db_bins[1],
                                         num_bins + 1, endpoint=True)
    
    fig.ppsd.cmap = obspy.imaging.cm.pqlx
    
    fig.ppsd.meshgrid = np.meshgrid(xedges, _db_bin_edges)
    ppsd = ax.pcolormesh(
        fig.ppsd.meshgrid[0], fig.ppsd.meshgrid[1], data.T,
        cmap=fig.ppsd.cmap, zorder=-1)
    fig.ppsd.quadmesh = ppsd
    
    ######
    percentiles=[10, 25, 50, 75, 90]
    percentiles=[50]
    percentiles=np.arange(5,100,15)
    #percentiles=[10, 50, 90]
    show_percentiles=True
    if show_percentiles:
        # for every period look up the approximate place of the percentiles
        for percentile in percentiles:
            periods, percentile_values = \
                get_percentile(hist_stack_cumul, percentile, _db_bin_edges, period_bin_centers)
            xdata = periods
            ax.plot(xdata, percentile_values, color="black", zorder=8, lw=1)
    ##########
    
    fig.ppsd.max_percentage = data.max()
    if fig.ppsd.max_percentage is not None:
                color_limits = (0, fig.ppsd.max_percentage)
                fig.ppsd.color_limits = color_limits
    if fig.ppsd.max_percentage is not None:
        ppsd.set_clim(*fig.ppsd.color_limits)
    label = "Probability"
    fig.ppsd.label = label
    if "colorbar" not in fig.ppsd:
        cb = plt.colorbar(ppsd, ax=ax)
        cb.mappable.set_clim(*fig.ppsd.color_limits)
        cb.set_label(fig.ppsd.label)
        fig.ppsd.colorbar = cb
    fig.ppsd.grid = True
    if fig.ppsd.grid:
        if fig.ppsd.cmap.name == "viridis":
            color = {"color": "0.7"}
        else:
            color = {}
        ax.grid(True, which="major", **color)
        ax.grid(True, which="minor", **color)
    
    #ax.set_xlim(*xlim)
    xaxis_frequency=False
    if xaxis_frequency:
        ax.set_xlabel('Frequency [Hz]')
        ax.invert_xaxis()
    else:
        ax.set_xlabel('Period [s]')
    ax.set_xscale('log')
    period_lim=(2/5, 100)
    ax.set_xlim(period_lim)
    ax.set_ylim(_db_bin_edges[0], _db_bin_edges[-1])
    
    ax.set_ylabel('Amplitude [$m^2/s^2/Hz$] [dB]')
    fig.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300)
    
    elif draw:
        with np.errstate(under="ignore"):
            plt.draw()
    #Epsd=get_percentile(hist_stack_cumul, 50, _db_bin_edges, period_bin_centers)

    return fig, _db_bin_edges, period_bin_centers, percentiles


def _plot_histogram_RX(Eppsd, hist_stack,hist_stack_cumul, db_bins,draw=False, filename=None, pltcolor='tab:green'):
    """
    Reuse a previously created figure returned by `plot(show=False)`
    and plot the current histogram stack (pre-computed using
    :meth:`calculate_histogram()`) into the figure. If a filename is
    provided, the figure will be saved to a local file.
    Note that many aspects of the plot are statically set during the first
    :meth:`plot()` call, so this routine can only be used to update with
    data from a new stack.
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4,4))
    fig.ppsd = AttribDict()
    ax = fig.add_subplot(111)
    ax = fig.axes[0]
    xlim = ax.get_xlim()
    '''if "quadmesh" in fig.ppsd:
        fig.ppsd.pop("quadmesh").remove()

    if fig.ppsd.cumulative:
        data = hist_stack_cumul * 100.0
    else:
        # avoid divison with zero in case of empty stack
        data = (
            hist_stack * 100.0 /
            (hist_stack.sum(axis=1)[0] or 1))
    '''   
    data = (
            hist_stack * 100.0 /
            (hist_stack.sum(axis=1)[0] or 1))

    _period_binning = _setup_period_binning(Eppsd)
    period_bin_centers=Eppsd.period_bin_centers
    
    xedges = np.concatenate([_period_binning[1, 0:1],
                               _period_binning[3, :]])
    #if fig.ppsd.xaxis_frequency:
    #    xedges = 1.0 / xedges
    
    # setup db binning
    # Set up the binning for the db scale.
    num_bins = int((db_bins[1] - db_bins[0]) / db_bins[2])
    _db_bin_edges = np.linspace(db_bins[0], db_bins[1],
                                         num_bins + 1, endpoint=True)
    
    fig.ppsd.cmap = obspy.imaging.cm.pqlx
    
    
    fig.ppsd.meshgrid = np.meshgrid(xedges, _db_bin_edges)
    ppsd = ax.pcolormesh(
        fig.ppsd.meshgrid[0], fig.ppsd.meshgrid[1], data.T,
        cmap=fig.ppsd.cmap, zorder=-1, alpha=0)
    fig.ppsd.quadmesh = ppsd
    
    ######
    percentiles=[10, 25, 50, 75, 90]
    percentiles=[50]
    #percentiles=np.arange(5,100,15)
    #percentiles=[10, 50, 90]
    show_percentiles=True
    
    periods, percentile_values = \
        get_percentile(hist_stack_cumul, 50, _db_bin_edges, period_bin_centers)
    xdata = periods
    ax.plot(xdata, percentile_values, color=pltcolor, zorder=8, lw=4)
    
    periods, percentile_values_l = \
        get_percentile(hist_stack_cumul, 25, _db_bin_edges, period_bin_centers)
    xdata = periods
    
    
    periods, percentile_values_h = \
        get_percentile(hist_stack_cumul, 75, _db_bin_edges, period_bin_centers)
    xdata = periods
    ax.fill_between(xdata, percentile_values_l, percentile_values_h, color=pltcolor, alpha=.25)
    ##########
    
    fig.ppsd.max_percentage = data.max()
    if fig.ppsd.max_percentage is not None:
                color_limits = (0, fig.ppsd.max_percentage)
                fig.ppsd.color_limits = color_limits
    if fig.ppsd.max_percentage is not None:
        ppsd.set_clim(*fig.ppsd.color_limits)
    label = "Probability [%]"
    fig.ppsd.label = label
    '''
    if "colorbar" not in fig.ppsd:
        cb = plt.colorbar(ppsd, ax=ax)
        cb.mappable.set_clim(*fig.ppsd.color_limits)
        cb.set_label(fig.ppsd.label)
        fig.ppsd.colorbar = cb
    '''
    fig.ppsd.grid = True
    if fig.ppsd.grid:
        if fig.ppsd.cmap.name == "viridis":
            color = {"color": "0.7"}
        else:
            color = {}
        ax.grid(True, which="major", **color)
        ax.grid(True, which="minor", **color)
    
    #ax.set_xlim(*xlim)
    xaxis_frequency=False
    if xaxis_frequency:
        ax.set_xlabel('Frequency [Hz]')
        ax.invert_xaxis()
    else:
        ax.set_xlabel('Period [s]')
    ax.set_xscale('log')
    period_lim=(2/5, 100)
    ax.set_xlim(period_lim)
    ax.set_ylim(_db_bin_edges[0], _db_bin_edges[-1])
    
    ax.set_ylabel('Amplitude [$m^2/s^2/Hz$] [dB]')
    
    
    #Set reference lines
    #ax.plot([2,600],[-11,-11],'k')
    #ax.plot([2/5,100],[-20,-20],'k')
    #ax.plot([2/5,100],[-26,-26],'k')
    ax.plot([2/5,100],[-40,-40],'k')
    ax.plot([2/5,100],[-60,-60],'k')

    bbox = dict(boxstyle="round", fc="0.8")
    #ax.annotate('20cm/s',xy=(2.3,-11),bbox=bbox)
    #ax.annotate('10cm/s',xy=(2.3,-18),bbox=bbox)
    #ax.annotate('5cm/s',xy=(2.3,-25),bbox=bbox)
    ax.annotate('1cm/s',xy=(1,-40),bbox=bbox)
    ax.annotate('0.1cm/s',xy=(1,-60),bbox=bbox)
    
    fig.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300)
    
    elif draw:
        with np.errstate(under="ignore"):
            plt.draw()
    #Epsd=get_percentile(hist_stack_cumul, 50, _db_bin_edges, period_bin_centers)

    return fig, _db_bin_edges, period_bin_centers, percentiles
