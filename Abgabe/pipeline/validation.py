"""Validation helpers for EEG sessions.

Dieses Modul bietet einfache Qualitäts-/Sanity-Checks für geladene EEG-Aufzeichnungen
und aggregiert Ergebnisse je Teilnehmer. Es ist bewusst leichtgewichtig gehalten und
vermeidet tiefe statistische Analysen (kann später ergänzt werden).

Checks (aktuell):
    * Sampling Rate Untergrenze
    * Mindestanzahl Kanäle
    * Mindest-/Höchstdauer
    * Grobe Amplituden-Heuristik & Nullsignal-Erkennung

Verbesserungsideen:
    * Kanalweise Varianz / Flatline-Detektion
    * Anteil saturierter Samples (Clipping)
    * Frequenzband-spezifische Power-Schwellen (Delta, Theta, Alpha ...)
    * Integration in MNE Report oder JSON-Export
"""

from dataclasses import dataclass
from typing import List, Dict, Any

from mne.io import Raw

# Relative Import, damit Modul innerhalb des Pakets funktioniert
from .data_loading import SessionData


@dataclass
class ValidationConfig:
    """Konfiguration für einfache Sanity-Checks.

    Parameters
    ----------
    min_sampling_rate:
        Minimal akzeptierte Abtastrate (Hz).
    min_channels:
        Mindestanzahl EEG-Kanäle für gültige Analyse.
    min_duration / max_duration:
        Zeitliche Schranken (Sekunden). Dauerberechnung erfolgt aus Raw.times.
    check_data_quality:
        Steuert zusätzliche Signal-Heuristiken (Nullsignal, extreme Amplitude).
    """

    min_sampling_rate: float = 100.0
    min_channels: int = 4
    min_duration: float = 60.0  # Sekunden
    max_duration: float = 3600.0  # 1 Stunde
    check_data_quality: bool = True


def validate_raw_data(raw: Raw, config: ValidationConfig) -> List[str]:
    """Prüfe eine einzelne Aufnahme und liefere Problem-Liste.

    Dauerberechnung: `raw.times[-1]` (≈ Gesamtzeit). Alternativ exakter:
    `(raw.n_times - 1) / raw.info['sfreq']` – für diese Heuristik ausreichend.
    """
    issues: List[str] = []

    # Sampling Rate
    if raw.info["sfreq"] < config.min_sampling_rate:
        issues.append(f"Low sampling rate ({raw.info['sfreq']} Hz)")

    # Kanalanzahl
    if len(raw.ch_names) < config.min_channels:
        issues.append(f"Too few channels ({len(raw.ch_names)})")

    # Dauer
    duration = raw.times[-1] if raw.n_times else 0.0
    if duration < config.min_duration:
        issues.append(f"Short recording ({duration:.1f}s)")
    elif duration > config.max_duration:
        issues.append(f"Very long recording ({duration:.1f}s)")

    # Einfache Qualitätsheuristik
    if config.check_data_quality:
        data = raw.get_data()
        if data.size == 0:
            issues.append("Empty data array")
        elif (data == 0).all():
            issues.append("All data points are zero")
        elif abs(data).max() > 1e-3:  # > 1 mV – grobe Heuristik
            issues.append("Unusually high amplitudes detected")

    return issues


def validate_session(
    session_data: SessionData, config: ValidationConfig = None
) -> Dict[str, List[str]]:
    """Validiere beide (optional vorhandenen) Sessions eines Teilnehmers."""
    if config is None:
        config = ValidationConfig()
    validation_results: Dict[str, List[str]] = {}
    for session_name, raw in (
        ("indoor", session_data.indoor_session),
        ("outdoor", session_data.outdoor_session),
    ):
        if raw is not None:
            issues = validate_raw_data(raw, config)
            if issues:
                validation_results[session_name] = issues
    return validation_results


def validate_all_sessions(
    sessions: List[SessionData], config: ValidationConfig = None
) -> Dict[str, Dict[str, List[str]]]:
    """Validiere mehrere Teilnehmer und sammle nur problematische Fälle."""
    if config is None:
        config = ValidationConfig()
    all_results: Dict[str, Dict[str, List[str]]] = {}
    for session_data in sessions:
        results = validate_session(session_data, config)
        if results:
            all_results[session_data.participant_name] = results
    return all_results


def print_validation_summary(
    validation_results: Dict[str, Dict[str, List[str]]],
) -> None:
    """Konsolenfreundliche Ausgabe aller Probleme (oder Erfolgsmeldung)."""
    if not validation_results:
        print("✓ Alle Sessions haben die Validierung bestanden!")
        return
    print("⚠ Validierungsprobleme gefunden:")
    for participant, sessions in validation_results.items():
        print(f"\n{participant}:")
        for session_name, issues in sessions.items():
            print(f"  {session_name}: {'; '.join(issues)}")


def get_validation_stats(sessions: List[SessionData]) -> Dict[str, Any]:
    """Aggregiere einfache Strukturstatistiken über alle geladenen Sessions."""
    total_participants = len(sessions)
    indoor_sessions = sum(1 for s in sessions if s.indoor_session is not None)
    outdoor_sessions = sum(1 for s in sessions if s.outdoor_session is not None)

    sampling_rates: List[float] = []
    channel_counts: List[int] = []
    for session in sessions:
        for raw in (session.indoor_session, session.outdoor_session):
            if raw is not None:
                sampling_rates.append(raw.info["sfreq"])
                channel_counts.append(len(raw.ch_names))

    return {
        "total_participants": total_participants,
        "indoor_sessions": indoor_sessions,
        "outdoor_sessions": outdoor_sessions,
        "participants": [s.participant_name for s in sessions],
        "sampling_rates": {
            "min": min(sampling_rates) if sampling_rates else 0,
            "max": max(sampling_rates) if sampling_rates else 0,
            "unique": list(set(sampling_rates)) if sampling_rates else [],
        },
        "channel_counts": {
            "min": min(channel_counts) if channel_counts else 0,
            "max": max(channel_counts) if channel_counts else 0,
            "unique": list(set(channel_counts)) if channel_counts else [],
        },
    }
