"""
EEG Validation Module

Dieses Modul enthält Funktionen zur Validierung von EEG-Daten.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from mne.io import Raw
from data_loading import SessionData


@dataclass
class ValidationConfig:
    """Konfiguration für die Datenvalidierung."""
    min_sampling_rate: float = 100.0
    min_channels: int = 4
    min_duration: float = 60.0  # Sekunden
    max_duration: float = 3600.0  # 1 Stunde
    check_data_quality: bool = True


def validate_raw_data(raw: Raw, config: ValidationConfig) -> List[str]:
    """Validiere eine einzelne Raw-Datei."""
    issues = []

    # Prüfe Sampling Rate
    if raw.info['sfreq'] < config.min_sampling_rate:
        issues.append(f"Low sampling rate ({raw.info['sfreq']} Hz)")

    # Prüfe Anzahl Kanäle
    if len(raw.ch_names) < config.min_channels:
        issues.append(f"Too few channels ({len(raw.ch_names)})")

    # Prüfe Dauer
    duration = raw.times[-1]
    if duration < config.min_duration:
        issues.append(f"Short recording ({duration:.1f}s)")
    elif duration > config.max_duration:
        issues.append(f"Very long recording ({duration:.1f}s)")

    # Prüfe Datenqualität
    if config.check_data_quality:
        data = raw.get_data()
        if (data == 0).all():
            issues.append("All data points are zero")
        elif abs(data).max() > 1e-3:  # > 1mV
            issues.append("Unusually high amplitudes detected")

    return issues


def validate_session(session_data: SessionData, config: ValidationConfig = None) -> Dict[str, List[str]]:
    """Validiere eine komplette Session."""
    if config is None:
        config = ValidationConfig()

    validation_results = {}

    for session_name, raw in [("indoor", session_data.indoor_session),
                              ("outdoor", session_data.outdoor_session)]:
        if raw is not None:
            issues = validate_raw_data(raw, config)
            if issues:
                validation_results[session_name] = issues

    return validation_results


def validate_all_sessions(sessions: List[SessionData], config: ValidationConfig = None) -> Dict[str, Dict[str, List[str]]]:
    """Validiere alle Sessions."""
    if config is None:
        config = ValidationConfig()

    all_results = {}

    for session_data in sessions:
        results = validate_session(session_data, config)
        if results:
            all_results[session_data.participant_name] = results

    return all_results


def print_validation_summary(validation_results: Dict[str, Dict[str, List[str]]]) -> None:
    """Drucke eine Zusammenfassung der Validierungsergebnisse."""
    if not validation_results:
        print("✓ Alle Sessions haben die Validierung bestanden!")
        return

    print("⚠ Validierungsprobleme gefunden:")
    for participant, sessions in validation_results.items():
        print(f"\n{participant}:")
        for session_name, issues in sessions.items():
            print(f"  {session_name}: {'; '.join(issues)}")


def get_validation_stats(sessions: List[SessionData]) -> Dict[str, Any]:
    """Erstelle Statistiken über die geladenen Sessions."""
    total_participants = len(sessions)
    indoor_sessions = sum(1 for s in sessions if s.indoor_session is not None)
    outdoor_sessions = sum(1 for s in sessions if s.outdoor_session is not None)

    # Sampling rates
    sampling_rates = []
    for session in sessions:
        for raw in [session.indoor_session, session.outdoor_session]:
            if raw is not None:
                sampling_rates.append(raw.info['sfreq'])

    # Channel counts
    channel_counts = []
    for session in sessions:
        for raw in [session.indoor_session, session.outdoor_session]:
            if raw is not None:
                channel_counts.append(len(raw.ch_names))

    return {
        "total_participants": total_participants,
        "indoor_sessions": indoor_sessions,
        "outdoor_sessions": outdoor_sessions,
        "participants": [s.participant_name for s in sessions],
        "sampling_rates": {
            "min": min(sampling_rates) if sampling_rates else 0,
            "max": max(sampling_rates) if sampling_rates else 0,
            "unique": list(set(sampling_rates)) if sampling_rates else []
        },
        "channel_counts": {
            "min": min(channel_counts) if channel_counts else 0,
            "max": max(channel_counts) if channel_counts else 0,
            "unique": list(set(channel_counts)) if channel_counts else []
        }
    }
