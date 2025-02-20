import random

import IPython.display as ipd  # for playing audios
import librosa  # for some audio processing
import librosa.display
import matplotlib.pyplot as plt  # matplot lib is the premiere plotting lib for Python: https://matplotlib.org/
import matplotlib.ticker as ticker
import numpy as np  # numpy is the premiere signal handling library for Python: http://www.numpy.org/
import scipy as sp  # for signal processing
from scipy import signal


# Starter Code
# The visualization code is provided to you for convenience
def plot_signal_to_axes(ax, s, sampling_rate, title=None, signal_label=None, marker=None):
    """Plots a sine wave s with the given sampling rate

    Parameters:
    ax: matplot axis to do the plotting
    s: numpy array
    sampling_rate: sampling rate of s
    title: chart title
    signal_label: the label of the signal
    """
    ax.plot(s, label=signal_label, marker=marker, alpha=0.9)
    ax.set(xlabel="Samples")
    ax.set(ylabel="Amplitude")
    if signal_label is not None:
        ax.legend()

    # we use y=1.14 to make room for the secondary x-axis
    # see: https://stackoverflow.com/questions/12750355/python-matplotlib-figure-title-overlaps-axes-label-when-using-twiny
    if title is not None:
        ax.set_title(title, y=1.1)

    ax.grid()

    # add in a secondary x-axis to draw the x ticks as time (rather than samples)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())

    ax_ticks = ax.get_xticks()[1:-1]
    ax2_tick_labels = ax.get_xticks()[1:-1] / sampling_rate

    num_samples_shown = ax.get_xlim()[1] - ax.get_xlim()[0]
    time_shown = num_samples_shown / sampling_rate
    if time_shown < 1:
        ax2.set_xlabel("Time (ms)")
        # format with 'g' causes insignificant trailing zeroes to be removed
        # https://stackoverflow.com/a/2440708 but also uses scientific notation, oh well!
        ax2_tick_labels = [f"{x * 1000:.1f}" for x in ax2_tick_labels]
    else:
        ax2.set_xlabel("Time (secs)")
        ax2_tick_labels = ["{:.2f}".format(x) for x in ax2_tick_labels]

    ax2.set_xticks(ax_ticks)
    ax2.set_xticklabels(ax2_tick_labels)


def plot_audio(s, sampling_rate, quantization_bits=16, title=None, xlim_zoom=None, highlight_zoom_area=True):
    """Calls plot_Signal but accepts quantization_bits"""
    plot_title = title
    if plot_title is None:
        plot_title = f"{quantization_bits}-bit, {sampling_rate} Hz audio"

    return plot_signal(
        s, sampling_rate, title=title, xlim_zoom=xlim_zoom, highlight_zoom_area=highlight_zoom_area
    )


def plot_signal(s, sampling_rate, title=None, xlim_zoom=None, highlight_zoom_area=True):
    """Plots time-series data with the given sampling_rate and xlim_zoom"""

    plot_title = title
    if plot_title is None:
        plot_title = f"Sampling rate: {sampling_rate} Hz"

    if xlim_zoom == None:
        fig, axes = plt.subplots(1, 1, figsize=(15, 6))

        plot_signal_to_axes(axes, s, sampling_rate, plot_title)
        return (fig, axes)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True, gridspec_kw={"width_ratios": [2, 1]})
        plot_signal_to_axes(axes[0], s, sampling_rate, plot_title)

        # if(xlim_zoom == None):
        #     xlim_zoom = get_random_xzoom(len(audio_data), 0.1)

        if highlight_zoom_area:
            # yellow highlight color: color='#FFFBCC'
            axes[0].axvspan(xlim_zoom[0], xlim_zoom[1], color="orange", alpha=0.3)

        axes[1].set_xlim(xlim_zoom)
        zoom_title = f"Signal zoomed: {int(xlim_zoom[0])} - {int(xlim_zoom[1])} samples"
        plot_signal_to_axes(axes[1], s, sampling_rate, zoom_title)
        axes[1].set_ylabel(None)
        fig.tight_layout()
        return (fig, axes)


def plot_spectrogram_to_axes(ax, s, sampling_rate, title=None, marker=None, custom_axes=True):
    """Plots a spectrogram wave s with the given sampling rate

    Parameters:
    ax: matplot axis to do the plotting
    s: numpy array
    sampling_rate: sampling rate of s
    title: chart title
    """

    specgram_return_data = ax.specgram(s, Fs=sampling_rate)

    # we use y=1.14 to make room for the secondary x-axis
    # see: https://stackoverflow.com/questions/12750355/python-matplotlib-figure-title-overlaps-axes-label-when-using-twiny
    if title is not None:
        ax.set_title(title, y=1.2)

    ax.set_ylabel("Frequency")

    # add in a secondary x-axis to draw the x ticks as time (rather than samples)
    if custom_axes:
        ax.set(xlabel="Samples")
        ax_xtick_labels = np.array(ax.get_xticks()) * sampling_rate
        ax_xtick_labels_strs = [f"{int(xtick_label)}" for xtick_label in ax_xtick_labels]
        ax.set_xticklabels(ax_xtick_labels_strs)

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xlabel("Time (secs)")
        ax2_tick_labels = ax_xtick_labels / sampling_rate
        ax2_tick_labels_strs = [f"{xtick_label:.1f}s" for xtick_label in ax2_tick_labels]
        ax2.set_xticks(ax.get_xticks())
        ax2.set_xticklabels(ax2_tick_labels_strs)
    return specgram_return_data


def remap(val, start1, stop1, start2, stop2):
    """Similar to Processing and Arduino's map function"""
    return ((val - start1) / (stop1 - start1)) * (stop2 - start2) + start2


def plot_spectrogram(s, sampling_rate, title=None, xlim_zoom=None, highlight_zoom_area=True):
    """Plots signal with the given sampling_Rate, quantization level, and xlim_zoom"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 4), gridspec_kw={"width_ratios": [2, 1]})

    if title is None:
        title = f"{len(s) * sampling_rate} sec Signal with {sampling_rate} Hz"

    specgram_return_data0 = plot_spectrogram_to_axes(axes[0], s, sampling_rate, title)

    if xlim_zoom == None:
        max_length = len(s)
        length = int(max_length * 0.1)
        random_start = random.randint(0, max_length - length)
        xlim_zoom = (random_start, random_start + length)

    axes[1].set_xlim(xlim_zoom)
    # axes[1].set_xlim(12000, 14000)
    specgram_return_data1 = plot_spectrogram_to_axes(
        axes[1], s, sampling_rate, title + " (Zoomed)", custom_axes=False
    )

    zoom_x1 = xlim_zoom[0] / sampling_rate
    zoom_x2 = xlim_zoom[1] / sampling_rate
    axes[1].set_xlim(zoom_x1, zoom_x2)  # but this one seems to work

    ax2 = axes[1].twiny()
    ax2.set_xlim(axes[1].get_xlim())
    ax2.set_xticks(axes[1].get_xticks())
    ax2_tick_labels_strs = [f"{xtick_label:.1f}s" for xtick_label in axes[1].get_xticks()]
    ax2.set_xticklabels(ax2_tick_labels_strs)
    ax2.set_xlabel("Time (secs)")

    ax_xtick_labels = np.array(axes[1].get_xticks()) * sampling_rate
    ax2_tick_labels_strs = [f"{int(xtick_label)}" for xtick_label in ax_xtick_labels]
    axes[1].set(xlabel="Samples")
    axes[1].set_xticklabels(ax2_tick_labels_strs)

    if highlight_zoom_area:
        # yellow highlight color: color='#FFFBCC'
        axes[0].axvline(x=zoom_x1, linewidth=2, color="r", alpha=0.8, linestyle="-.")
        axes[0].axvline(x=zoom_x2, linewidth=2, color="r", alpha=0.8, linestyle="-.")

    fig.tight_layout()
    return (fig, axes, specgram_return_data0, specgram_return_data1)


def plot_signal_and_spectrogram(
    s, sampling_rate, quantization_bits, xlim_zoom=None, highlight_zoom_area=True
):
    """Plot waveforms and spectrograms together"""
    fig = plt.figure(figsize=(15, 9))
    spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=[2, 1], height_ratios=[1, 1])
    plot_title = f"{quantization_bits}-bit, {sampling_rate} Hz audio"

    ax_waveform1 = plt.subplot(spec[0, 0])
    ax_waveform1.set_xlim(0, len(s))
    ax_waveform2 = plt.subplot(spec[0, 1], sharey=ax_waveform1)

    ax_spectrogram1 = plt.subplot(spec[1, 0])
    ax_spectrogram2 = plt.subplot(spec[1, 1])

    plot_signal_to_axes(ax_waveform1, s, sampling_rate, plot_title)
    specgram_return_data = plot_spectrogram_to_axes(ax_spectrogram1, s, sampling_rate, plot_title)
    # print(len(specgram_return_data[2]))

    # print(ax_waveform1.get_xlim())
    # print(ax_spectrogram1.get_xlim())
    waveform_xrange = ax_waveform1.get_xlim()[1] - ax_waveform1.get_xlim()[0]

    ax_waveform2.set_xlim(xlim_zoom)
    plot_signal_to_axes(ax_waveform2, s, sampling_rate, plot_title + " zoomed")

    zoom_x1 = remap(
        xlim_zoom[0],
        ax_waveform1.get_xlim()[0],
        ax_waveform1.get_xlim()[1],
        ax_spectrogram1.get_xlim()[0],
        ax_spectrogram1.get_xlim()[1],
    )
    zoom_x2 = remap(
        xlim_zoom[1],
        ax_waveform1.get_xlim()[0],
        ax_waveform1.get_xlim()[1],
        ax_spectrogram1.get_xlim()[0],
        ax_spectrogram1.get_xlim()[1],
    )

    # print(ax_spectrogram2.get_xlim(), zoom_x1, zoom_x2)
    ax_spectrogram2.set_xlim(zoom_x1, zoom_x2)  # this won't make a difference
    plot_spectrogram_to_axes(ax_spectrogram2, s, sampling_rate, plot_title, custom_axes=False)
    ax_spectrogram2.set_xlim(zoom_x1, zoom_x2)  # but this one seems to work

    ax2 = ax_spectrogram2.twiny()
    ax2.set_xlim(ax_spectrogram2.get_xlim())
    ax2.set_xticks(ax_spectrogram2.get_xticks())
    ax2_tick_labels_strs = [f"{xtick_label:.2f}s" for xtick_label in ax_spectrogram2.get_xticks()]
    ax2.set_xticklabels(ax2_tick_labels_strs)
    ax2.set_xlabel("Time (secs)")

    ax_xtick_labels = np.array(ax_spectrogram2.get_xticks()) * sampling_rate
    ax2_tick_labels_strs = [f"{int(xtick_label)}" for xtick_label in ax_xtick_labels]
    ax_spectrogram2.set(xlabel="Samples")
    ax_spectrogram2.set_xticks(ax_spectrogram2.get_xticks())
    ax_spectrogram2.set_xticklabels(ax2_tick_labels_strs)

    if highlight_zoom_area:
        # yellow highlight color: color='#FFFBCC'
        ax_waveform1.axvspan(xlim_zoom[0], xlim_zoom[1], color="orange", alpha=0.3)
        ax_spectrogram1.axvline(x=zoom_x1, linewidth=2, color="r", alpha=0.8, linestyle="-.")
        ax_spectrogram1.axvline(x=zoom_x2, linewidth=2, color="r", alpha=0.8, linestyle="-.")

    fig.tight_layout()
