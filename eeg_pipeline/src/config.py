"""Configuration classes for the EEG workload decoding pipeline.

This module defines dataclasses that group together the key hyper‑
parameters of the preprocessing, feature extraction and model training
stages. Keeping configuration values in dedicated classes makes it
straightforward to adjust the pipeline without touching the core logic.

The default values reflect commonly used settings in mobile EEG
workload research: 50 Hz notch filtering to remove power line noise, a
1–40 Hz bandpass to isolate the theta, alpha and beta bands【102020619966441†L182-L209】,
and two second windows with 50 % overlap for spectral features. ERP
features are extracted from epochs ranging from −0.2 s to 0.8 s around
stimulus onset, and the P300 window is set to 250–450 ms based on the
typical latency of the P300 component【990978214535399†L150-L155】. Logistic
regression with elastic‑net regularisation is used as the default
classifier; linear support vector machines with Platt scaling can be
selected via the ``model_type`` field.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class PreprocessingConfig:
    """Parameters controlling the EEG preprocessing stage.

    Attributes
    ----------
    notch_freq : float
        Frequency at which to apply a notch filter to remove mains
        interference. For power systems in Europe this is 50 Hz
        【102020619966441†L182-L209】.
    l_freq : float
        Low cut‑off frequency for high‑pass filtering. Setting this to
        1 Hz removes slow drifts without excessively distorting ERP
        waveforms【102020619966441†L182-L209】.
    h_freq : float
        High cut‑off frequency for low‑pass filtering. A cut‑off of
        40 Hz preserves the bulk of the task‑relevant oscillations while
        suppressing high frequency noise and muscle artifacts【102020619966441†L182-L209】.
    resample : float | None
        Target sampling rate in Hz. If ``None`` the data are not
        resampled. Down‑sampling can reduce computational load.
    reference : str | list[str]
        Re‑reference strategy. Use ``'average'`` for a virtual average
        reference【296520584149078†L465-L477】 or provide a list of channel names to
        compute a custom reference (e.g. mastoid electrodes).
    zscore_per_subject : bool
        If ``True``, z‑score normalisation is applied per subject to
        mitigate inter‑subject amplitude differences【473351643201165†L170-L183】.
    """

    notch_freq: float = 50.0
    l_freq: float = 1.0
    h_freq: float = 40.0
    resample: float | None = None
    reference: str | tuple[str, ...] = 'average'
    zscore_per_subject: bool = True


@dataclass
class FeatureConfig:
    """Parameters controlling feature extraction.

    Attributes
    ----------
    window_length : float
        Length of the sliding windows in seconds. Short windows (2 s by
        default) with 50 % overlap capture transient changes in the
        spectral power while keeping enough samples for reliable
        estimation.
    window_overlap : float
        Fractional overlap between consecutive windows.
    bands : dict[str, tuple[float, float]]
        Mapping of band names to (low, high) frequency boundaries in Hz.
        The default configuration covers the theta (4–7 Hz), alpha
        (8–12 Hz) and beta (13–30 Hz) bands that have been shown to
        correlate with mental workload【634168958309391†L324-L340】.
    erp_interval : tuple[float, float]
        Time window relative to stimulus onset used to extract ERP
        segments. By default epochs span from −0.2 s pre‑stimulus to
        0.8 s post‑stimulus.
    p300_window : tuple[float, float]
        Latency window (in seconds after stimulus) for measuring P300
        amplitude. P300 peaks typically occur between 250 and 500 ms
        post‑stimulus【990978214535399†L150-L155】.
    """

    window_length: float = 2.0
    window_overlap: float = 0.5
    bands: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            'theta': (4.0, 7.0),
            'alpha': (8.0, 12.0),
            'beta': (13.0, 30.0),
        }
    )
    erp_interval: Tuple[float, float] = (-0.2, 0.8)
    p300_window: Tuple[float, float] = (0.25, 0.45)


@dataclass
class ModelConfig:
    """Parameters for the linear classifier.

    Attributes
    ----------
    model_type : str
        Type of linear model to train. Either ``'logreg'`` for
        logistic regression or ``'svm'`` for a linear support vector
        machine. Models based on linear decision boundaries are robust
        and widely used in EEG mental workload research【634168958309391†L160-L168】.
    penalty : str
        Penalty used by logistic regression. Supported values are
        ``'l1'``, ``'l2'`` or ``'elasticnet'``.
    l1_ratio : float
        Elastic‑net mixing parameter. Ignored unless ``penalty`` is
        ``'elasticnet'``.
    C : float
        Inverse of regularisation strength for both logistic regression
        and SVM.
    random_state : int
        Random seed used for reproducibility.
    """

    model_type: str = 'logreg'
    penalty: str = 'elasticnet'
    l1_ratio: float = 0.5
    C: float = 1.0
    random_state: int = 42


@dataclass
class TrainingConfig:
    """Cross‑validation and training options.

    Attributes
    ----------
    cv_strategy : str
        Cross‑validation scheme. The default ``'leave-one-subject-out'``
        trains on all but one subject and tests on the held‑out subject.
    n_jobs : int
        Number of parallel jobs to use for model training and
        evaluation. Use ``-1`` to utilise all available cores.
    """

    cv_strategy: str = 'leave-one-subject-out'
    n_jobs: int = -1


@dataclass
class EEGNetConfig:
    """Configuration for the optional EEGNet baseline.

    Attributes
    ----------
    sampling_rate : float
        Sampling rate of the input signals in Hz.
    input_channels : int
        Number of spatial channels fed to the network.
    input_length : int | None
        Length of the temporal dimension in samples. If ``None`` it
        should be computed from the window length and the sampling rate.
    """

    sampling_rate: float = 128.0
    input_channels: int = 8
    input_length: int | None = None