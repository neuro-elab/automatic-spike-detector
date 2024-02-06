import os
import re
from datetime import datetime, timedelta
from typing import List, Sequence, Dict

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from loguru import logger

from spidet.utils.variables import LEAD_PREFIXES_EL010

mpl.use("agg")


def plot_std_line_length(
    experiment_dir: str,
    std_line_length: np.ndarray,
    start_time_recording: datetime,
    display_all: bool = False,
    offset: timedelta = timedelta(),
    duration: int = 10,
    sfreq: float = 50,
    seizure: int = None,
) -> None:
    dir_path = (
        os.path.join(experiment_dir, "plots_line_length_data", "std_line_length")
        if seizure is None
        else os.path.join(
            experiment_dir,
            "plots_line_length_data",
            "std_line_length",
            f"seizure_{seizure}",
        )
    )
    os.makedirs(dir_path, exist_ok=True)

    offset_seconds = int(offset.total_seconds())
    if display_all and offset_seconds != 0:
        logger.warning("display_all is True, ignoring any given offset and duration")
        offset_seconds = 0

    # Determine beginning of the sub-period to plot the data for
    start_time_display_period: datetime = start_time_recording + timedelta(
        seconds=offset_seconds
    )

    # Create file TITLE
    period = "all" if display_all else f"{duration}s"
    title = create_file_title(
        exp_dir=experiment_dir,
        data_kind="Std Line Length",
        start_time_recording=start_time_recording,
        start_time_display_period=start_time_display_period,
        offset_seconds=offset_seconds,
        period=period,
    )

    # Start and end of the time period to display data for
    start = int(sfreq * offset_seconds)
    stop = len(std_line_length) if display_all else start + int(sfreq * duration)

    # Extract sub-period from preprocessed
    # _eeg
    ll_period = std_line_length[start:stop]

    # Figure to plot brain regions combined in the same file
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))

    # Generate xticks, label as time of the day
    xticks = np.linspace(start=0, stop=len(ll_period), num=11)
    ticks_as_datetime = [
        (
            start_time_recording + timedelta(seconds=offset_seconds + tick / sfreq)
        ).strftime("%T.%f")[:-4]
        for tick in xticks
    ]

    # Plot
    ax.plot(ll_period.T)
    ax.legend(["Std Line Length"], loc="center left")
    ax.set_xticks(xticks, ticks_as_datetime)
    ax.set_xlabel("Time of the day [HH:MM:SS.ff]")
    ax.set_ylabel("Volt")

    fig.suptitle(title)
    fig.savefig(
        os.path.join(
            dir_path,
            create_filename(
                prefix="Std_LL",
                offset=offset_seconds,
                period=period,
                seizure=seizure,
            ),
        )
    )


def plot_line_length_data(
    experiment_dir: str,
    line_length_eeg: np.ndarray,
    channel_names: List[str],
    start_time_recording: datetime,
    lead_prefixes: Sequence[str] = LEAD_PREFIXES_EL010,
    display_all: bool = False,
    offset: timedelta = timedelta(),
    duration: int = 10,
    sfreq: float = 50,
    y_lim: float = None,
    seizure: int = None,
    spike_annotations: List[datetime] = None,
) -> None:
    dir_path = (
        os.path.join(experiment_dir, "plots_line_length_data")
        if seizure is None
        else os.path.join(
            experiment_dir, "plots_line_length_data", f"seizure_{seizure}"
        )
    )
    os.makedirs(dir_path, exist_ok=True)

    offset_seconds = int(offset.total_seconds())
    if display_all and offset_seconds != 0:
        logger.warning("display_all is True, ignoring any given offset and duration")
        offset_seconds = 0

    # Determine beginning of the sub-period to plot the data for
    start_time_display_period: datetime = start_time_recording + timedelta(
        seconds=offset_seconds
    )

    # Create file TITLE
    period = "all" if display_all else f"{duration}s"
    title = create_file_title(
        exp_dir=experiment_dir,
        data_kind="LL EEG",
        start_time_recording=start_time_recording,
        start_time_display_period=start_time_display_period,
        offset_seconds=offset_seconds,
        period=period,
    )

    # Start and end of the time period to display data for
    start = int(sfreq * offset_seconds)
    stop = line_length_eeg.shape[1] if display_all else start + int(sfreq * duration)

    # Extract sub-period from preprocessed_eeg
    eeg_period = line_length_eeg[:, start:stop]

    # Figure to plot brain regions combined in the same file
    fig_comb, ax_comb = plt.subplots(len(lead_prefixes), 1, figsize=(20, 20))

    for idx, prefix in enumerate(lead_prefixes):
        logger.debug(
            seizure_prefix(
                f"LL EEG {prefix}: generate plot for start time {start_time_display_period.time()} and duration {duration} seconds",
                seizure,
            )
        )

        # Figure for separate plot
        fig, ax = plt.subplots(figsize=(20, 20))

        # Extract channels for particular prefix
        channels = list(
            filter(lambda channel_name: channel_name.startswith(prefix), channel_names)
        )
        channels_idx_start = channel_names.index(channels[0])
        channels_idx_stop = channel_names.index(channels[-1])

        if spike_annotations is not None:
            channels.insert(0, "Spikes")

        # Generate xticks, label as time of the day
        xticks = np.linspace(start=0, stop=eeg_period.shape[1], num=11)
        ticks_as_datetime = [
            (
                start_time_recording + timedelta(seconds=offset_seconds + tick / sfreq)
            ).strftime("%T.%f")[:-4]
            for tick in xticks
        ]

        # Map spike annotations, if available, to x-axis
        if spike_annotations is not None:
            spikes = []
            for spike in spike_annotations:
                offset_spike = spike.timestamp() - start_time_recording.timestamp()
                if display_all:
                    spikes.append(offset_spike * sfreq)
                elif offset_seconds < offset_spike < offset_seconds + duration:
                    offset_within_period = offset_spike - offset_seconds
                    spikes.append(offset_within_period * sfreq)
            ax_comb[idx].vlines(
                spikes,
                0,
                np.max(eeg_period[channels_idx_start:channels_idx_stop, :].T),
                linestyles="solid",
                colors="silver",
            )
            ax.vlines(
                spikes,
                0,
                np.max(eeg_period[channels_idx_start:channels_idx_stop, :].T),
                linestyles="solid",
                colors="silver",
            )

        # Plot in separate file
        ax.plot(eeg_period[channels_idx_start:channels_idx_stop, :].T)
        ax.legend(channels, loc="center left")
        ax.set_xticks(xticks, ticks_as_datetime)
        ax.set_xlabel("Time of the day [HH:MM:SS.ff]")
        ax.set_ylabel("Volt")
        if y_lim is not None:
            ax.set_ylim(top=y_lim)

        # Plot in common file
        ax_comb[idx].plot(eeg_period[channels_idx_start:channels_idx_stop, :].T)
        ax_comb[idx].set_xticks(xticks, ticks_as_datetime)
        ax_comb[idx].set_xlabel("Time of the day [HH:MM:SS.ff]")
        ax_comb[idx].set_ylabel("Volt")
        ax_comb[idx].legend(
            channels, loc=("center right" if idx % 2 == 0 else "center left")
        )

        # Create directory for prefix if it does not already exist
        os.makedirs(
            os.path.join(dir_path, prefix),
            exist_ok=True,
        )

        fig.suptitle(title)
        fig.savefig(
            os.path.join(
                dir_path,
                prefix,
                create_filename(
                    prefix=prefix, offset=offset_seconds, period=period, seizure=seizure
                ),
            )
        )

    filename_prefix = (
        "EEG_LL" if lead_prefixes == LEAD_PREFIXES_EL010 else "_".join(lead_prefixes)
    )
    fig_comb.subplots_adjust(hspace=0.6)
    fig_comb.suptitle(title)
    fig_comb.savefig(
        os.path.join(
            dir_path,
            create_filename(
                prefix=filename_prefix,
                offset=offset_seconds,
                period=period,
                seizure=seizure,
            ),
        ),
    )


def plot_w_and_consensus_matrix(
    w_matrices: list[np.ndarray],
    consensus_matrices: List[np.ndarray],
    experiment_dir: str,
    channel_names: List[str],
    rank_labels_idx: Dict[int, Dict[int, str]] = None,
) -> None:
    first_rank = w_matrices[0].shape[1]

    nr_ranks = len(w_matrices)

    nr_cols = 3 if nr_ranks >= 9 else 2
    nr_rows = int(
        nr_ranks / nr_cols
        if nr_ranks % nr_cols == 0
        else (nr_ranks + nr_ranks % nr_cols) / nr_cols
    )

    fig_w, ax_w = plt.subplots(nr_rows, nr_cols, figsize=(20, 20))
    fig_consensus, ax_consensus = plt.subplots(nr_rows, nr_cols, figsize=(20, 20))

    nr_ranks_plotted = 0
    for row in range(nr_rows):
        for col in range(nr_cols):
            if nr_ranks_plotted >= nr_ranks:
                break
            # PLot W matrix
            w_best = w_matrices[nr_ranks_plotted]
            ax_w[row, col].imshow(w_best, cmap=mpl.colormaps["YlOrRd"], aspect="auto")
            ax_w[row, col].set_title(f"Rank = {nr_ranks_plotted + first_rank}")

            labels = (
                dict()
                if rank_labels_idx is None
                else rank_labels_idx.get(nr_ranks_plotted + first_rank)
            )
            xticks_labels = [
                f"W{rank + 1}" if labels is None else labels.get(rank, f"W{rank + 1}")
                for rank in range(nr_ranks_plotted + first_rank)
            ]
            ax_w[row, col].set_xticks(range(nr_ranks_plotted + first_rank))
            ax_w[row, col].set_xticklabels(xticks_labels, fontsize=6)
            ax_w[row, col].set_yticks(range(len(channel_names)))
            ax_w[row, col].set_yticklabels(channel_names, fontsize=3)
            ax_w[row, col].tick_params(bottom=False, top=False, left=False)

            # Plot consensus matrix
            consensus_matrix = consensus_matrices[nr_ranks_plotted]
            ax_consensus[row, col].matshow(
                consensus_matrix, cmap=mpl.colormaps["YlGn"], aspect="auto"
            )
            ax_consensus[row, col].set_title(f"Rank = {nr_ranks_plotted + first_rank}")

            ax_consensus[row, col].set_xticks(range(len(channel_names)))
            ax_consensus[row, col].set_xticklabels(
                channel_names, fontsize=3, rotation=90
            )
            ax_consensus[row, col].set_yticks(range(len(channel_names)))
            ax_consensus[row, col].set_yticklabels(channel_names, fontsize=3)
            ax_consensus[row, col].tick_params(bottom=False, top=False, left=False)

            nr_ranks_plotted += 1

    # Delete unused subplots
    if nr_rows * nr_cols >= nr_ranks_plotted:
        nr_deletes = nr_rows * nr_cols - nr_ranks_plotted
        for col in range(1, nr_deletes + 1):
            fig_w.delaxes(ax_w[-1, -col])
            fig_consensus.delaxes(ax_consensus[-1, -col])

    file_label = extract_label_from_path(experiment_dir)

    fig_w.suptitle(f"{file_label} - W matrix")
    fig_w.subplots_adjust(hspace=0.3, wspace=0.3)
    fig_w.colorbar(
        mpl.cm.ScalarMappable(cmap=mpl.colormaps["YlOrRd"]), ax=ax_w, shrink=0.5
    )
    fig_w.savefig(experiment_dir + "/W_matrix.pdf")

    fig_consensus.suptitle(f"{file_label} - CONSENSUS matrix")
    fig_consensus.subplots_adjust(hspace=0.5, wspace=0.3)
    fig_consensus.colorbar(
        mpl.cm.ScalarMappable(cmap=mpl.colormaps["YlGn"]), ax=ax_consensus, shrink=0.5
    )
    fig_consensus.savefig(experiment_dir + "/consensus_matrix.pdf")


def plot_h_matrix_period(
    experiment_dir: str,
    h_matrices: List[np.ndarray],
    start_time_recording: datetime,
    display_all: bool = False,
    offset: timedelta = timedelta(),
    duration: int = 10,
    sfreq: float = 50,
    seizure: int = None,
    rank_labels_idx: Dict[int, Dict[int, str]] = None,
    spike_annotations: List[datetime] = None,
) -> None:
    rank_dirs = get_rank_dirs_sorted(experiment_dir)
    dir_path = (
        os.path.join(experiment_dir, "plots_h_matrix")
        if seizure is None
        else os.path.join(experiment_dir, "plots_h_matrix", f"seizure_{seizure}")
    )
    os.makedirs(dir_path, exist_ok=True)

    offset_seconds = int(offset.total_seconds())
    if display_all and offset_seconds != 0:
        logger.warning("display_all is True, ignoring any given offset and duration")
        offset_seconds = 0

    start_time_display_period: datetime = start_time_recording + timedelta(
        seconds=offset_seconds
    )

    fig, ax = plt.subplots(len(rank_dirs), 1, figsize=(20, 20))

    logger.debug(
        seizure_prefix(
            f"H MATRIX: generate plot for start time {start_time_display_period.time()} and duration {duration} seconds",
            seizure,
        )
    )

    for idx in range(len(rank_dirs)):
        current_rank_dir = rank_dirs[idx]
        current_rank = int(current_rank_dir[current_rank_dir.rfind("=") + 1 :])
        labels = (
            dict() if rank_labels_idx is None else rank_labels_idx.get(current_rank)
        )
        labels = [
            f"H{rank + 1}" if labels is None else labels.get(rank, f"H{rank + 1}")
            for rank in range(current_rank)
        ]
        if spike_annotations is not None:
            labels.insert(0, "Spikes")

        h_best = h_matrices[idx]

        # Start and end of the time period to display data for
        if display_all:
            start = 0
            stop = h_best.shape[1]
        else:
            start = int(sfreq * offset_seconds)
            stop = start + int(sfreq * duration)

        # Extract sub-period from H
        h_period = h_best[:, start:stop]

        # Map spike annotations, if available, to x-axis
        if spike_annotations is not None:
            spikes = []
            for spike in spike_annotations:
                offset_spike = spike.timestamp() - start_time_recording.timestamp()
                if display_all:
                    spikes.append(offset_spike * sfreq)
                elif offset_seconds < offset_spike < offset_seconds + duration:
                    offset_within_period = offset_spike - offset_seconds
                    spikes.append(offset_within_period * sfreq)
            ax[idx].vlines(
                spikes, 0, np.max(h_period), linestyles="solid", colors="silver"
            )

        # Plot
        ax[idx].plot(h_period.T)
        ax[idx].legend(labels, loc=("center right" if idx % 2 == 0 else "center left"))
        xticks = np.linspace(start=0, stop=h_period.shape[1], num=11)

        # Labels for x-axis as time of the day of the recording
        ticks_as_datetime = [
            (
                start_time_recording + timedelta(seconds=offset_seconds + tick / sfreq)
            ).strftime("%T.%f")[:-4]
            for tick in xticks
        ]

        ax[idx].set_xticks(xticks, ticks_as_datetime)
        ax[idx].set_xlabel("Time of the day [HH:MM:SS.ff]")
        ax[idx].set_title(f"Rank = {current_rank}")

    fig.subplots_adjust(hspace=1.0)
    period = "all" if display_all else f"{duration}s"
    file_title = create_file_title(
        exp_dir=experiment_dir,
        data_kind="H matrix",
        start_time_recording=start_time_recording,
        start_time_display_period=start_time_display_period,
        offset_seconds=offset_seconds,
        period=period,
    )
    fig.suptitle(file_title)
    plt.savefig(
        os.path.join(
            dir_path,
            create_filename(
                prefix="H", offset=offset_seconds, period=period, seizure=seizure
            ),
        ),
    )


def plot_metrics(metrics: pd.DataFrame, dir_path: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    # Plot Cophenetic Correlation
    ax[0].plot(metrics["Rank"], metrics["Cophenetic Correlation"])
    ax[0].set_ylim(0.82, 0.93)
    ax[0].set_title("Cophenetic Correlation")
    ax[0].set_xlabel("Rank")

    # Plot delta of auc of cdf
    ax[1].plot(metrics["Rank"], metrics["delta_k (CDF)"])
    ax[1].set_title("Change AUC of CDF")
    ax[1].set_xlabel("Rank")
    ax[1].set_ylabel("Delta k")

    file_label = extract_label_from_path(dir_path)
    fig.suptitle(f"{file_label} - Metrics")

    plt.savefig(f"{dir_path}/metrics.pdf")


def get_rank_dirs_sorted(experiment_dir: str) -> List[str]:
    # Retrieve the paths to the rank directories within the experiment folder
    rank_dirs = [
        experiment_dir + "/" + k_dir
        for k_dir in os.listdir(experiment_dir)
        if os.path.isdir(os.path.join(experiment_dir, k_dir)) and "k=" in k_dir
    ]

    # Only ranks specified are considered
    ranks = ["k=2", "k=3", "k=4", "k=5", "k=6", "k=7", "k=8", "k=9", "k=10"]

    def dir_contains_rank(rank_dir: str) -> bool:
        for k in ranks:
            if k in rank_dir:
                return True
        return False

    rank_dirs = list(filter(lambda rank_dir: dir_contains_rank(rank_dir), rank_dirs))
    return sorted(rank_dirs, key=lambda x: int(re.search(r"\d+$", x).group()))


def extract_label_from_path(experiment_dir: str) -> str:
    start_idx = experiment_dir.rfind("/") + 1
    end_idx = start_idx + experiment_dir[start_idx:].find("_")
    return experiment_dir[start_idx:end_idx]


def create_file_title(
    exp_dir: str,
    data_kind: str,
    start_time_recording: datetime,
    start_time_display_period: datetime,
    offset_seconds: int,
    period: str,
) -> str:
    return (
        f"{extract_label_from_path(exp_dir)} - "
        f"{data_kind} - Start time: {start_time_display_period.time()}, "
        f"Period: {period} (Start of recording: {start_time_recording}, Offset: {offset_seconds} seconds)"
    )


def seizure_prefix(log_msg: str, seizure: int) -> str:
    return log_msg if seizure is None else f"SEIZURE {seizure} - {log_msg}"


def create_filename(prefix: str, offset: int, period: str, seizure: int) -> str:
    seizure = seizure if seizure is None else f"Seizure{seizure}"
    filename = "_".join(
        filter(None, [seizure, prefix, "offset", f"{offset}s", "period", period])
    )
    return f"{filename}.pdf"
