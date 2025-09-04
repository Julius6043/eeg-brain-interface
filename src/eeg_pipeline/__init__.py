"""EEG Processing Pipeline Package.

Dieses Paket stellt eine vollständige Pipeline für die Verarbeitung von EEG-Daten bereit,
einschließlich Laden, Preprocessing und Marker-basierte Annotationen.
"""

from .data_loading import DataLoadingConfig, SessionData, load_all_sessions
from .preprocessing import PreprocessingConfig, preprocess_raw
from .marker_annotation import annotate_raw_with_markers
from .pipeline import EEGPipeline, PipelineConfig, create_default_config

__all__ = [
    "DataLoadingConfig",
    "SessionData",
    "load_all_sessions",
    "PreprocessingConfig",
    "preprocess_raw",
    "annotate_raw_with_markers",
    "EEGPipeline",
    "PipelineConfig",
    "create_default_config",
]
