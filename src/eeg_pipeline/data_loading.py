"""Data Loading Utilities for EEG Pipeline.

Dieses Modul kapselt das Einlesen von XDF-Dateien (EEG + Marker) und
stellt strukturierte Objekte für nachgelagerte Verarbeitung bereit.

Designziele:
    * Robuste, fehlertolerante Loader (fangen Exceptions lokal ab).
    * Klare Trennung zwischen Konfiguration (`DataLoadingConfig`) und Datencontainer (`SessionData`).
    * Minimale Heuristik zur Stream-Selektion (EEG + Marker) – kann später erweitert werden.

Wichtige Annahmen / Grenzen:
    * Es wird aktuell nur EIN EEG-Stream und EIN Marker-Stream gewählt (falls mehrere vorhanden, überschreibt der letzte Treffer den vorherigen).
    * Marker-Datei (CSV) wird über ein generisches Pattern gesucht (erstes `*.csv` in einem übergeordneten Ordner) – könnte falsch greifen, falls mehrere CSVs existieren.
    * Kanäle werden generisch als EEG1..EEG<N> benannt (keine Original-Kanalnamen aus dem XDF-Metadaten-Tree extrahiert).
    * Autoskalierung interpretiert Werte mit Median(|x|) > 1e-3 als Mikrovolt und re-skaliert nach Volt.

Empfohlene zukünftige Verbesserungen (TODO-Hinweise im Code gesetzt):
    - Präzisere Kanalnamensextraktion aus Stream-Metadaten.
    - Selektions-Prioritätenliste für konkurrierende EEG-Streams.
    - Validierung / Logging über standardisiertes Logger-Interface statt `print`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import mne
import numpy as np
import pyxdf
import pandas as pd


@dataclass
class DataLoadingConfig:
    """Konfigurationsparameter für das Laden von XDF-Daten.

    Attribute
    ---------
    channels_keep:
        Explizite Liste von Kanalnamen, die behalten werden sollen. Falls `None`, wird
        auf die ersten `max_channels` reduziert.
    montage:
        Name einer in MNE bekannten Elektroden-Montage (z.B. "standard_1020").
    auto_scale_to_volts:
        Heuristische Skalierung der Rohdaten: Wenn Werte vermutlich in µV vorliegen,
        werden sie in Volt umgerechnet (Multiplikation mit 1e-6).
    max_channels:
        Fallback-Limit, falls `channels_keep` nicht gesetzt ist.
    """

    channels_keep: Optional[List[str]] = None
    montage: str = "standard_1020"
    auto_scale_to_volts: bool = True
    max_channels: int = 8


@dataclass
class SessionData:
    """Strukturierte Sammlung der (bis zu zwei) Sessions eines Teilnehmers.

    Die Pipeline unterscheidet aktuell zwischen einer Session `ses-S001` (hier
    als "indoor" bezeichnet) und `ses-S002` ("outdoor"). Diese semantische
    Zuordnung erfolgt über Ordnernamen – es gibt keine zusätzliche Validierung
    anhand externer Metadaten.
    """

    participant_name: str
    indoor_session: Optional[mne.io.Raw] = None
    indoor_markers: Optional[pd.DataFrame] = None
    outdoor_session: Optional[mne.io.Raw] = None
    outdoor_markers: Optional[pd.DataFrame] = None


def load_xdf_safe(path: Path) -> Tuple[Optional[list], Optional[dict]]:
    """Lade eine XDF-Datei robust.

    Rückgabe
    --------
    streams:
        Liste der extrahierten Streams oder `None` bei Fehler.
    header:
        Globale Headerinformationen des XDF Containers.

    Fehler werden abgefangen, geloggt (stdout) und führen NICHT zum Abbruch
    des Gesamtprozesses (Fail-soft Strategie).
    """
    try:
        streams, header = pyxdf.load_xdf(str(path))
        return streams, header
    except Exception as e:  # pragma: no cover - defensive I/O
        print(f"[WARN] XDF-Load fehlgeschlagen für {path}: {e}")
        return None, None


def _safe_get(d: dict, key: str, default):
    """Robuster Zugriff auf (verschachtelte) XDF-Metadaten.

    XDF-Felder liegen oft als Ein-Element-Listen vor. Diese Funktion reduziert
    solche Container automatisch und liefert bei Fehlern einen Defaultwert.
    """
    try:
        v = d.get(key, default)
        if isinstance(v, (list, tuple)) and len(v) == 1:
            return v[0]
        return v
    except Exception:  # pragma: no cover - defensive
        return default


def pick_streams(streams: list) -> Tuple[Optional[dict], Optional[dict]]:
    """Wähle heuristisch EEG- und Marker-Stream.

    Strategie (einfach):
        * EEG: letzter Stream dessen type ODER name 'eeg' enthält, oder Name 'unicorn'.
        * Marker: letzter Stream dessen type/name 'marker(s)' enthält.

    Hinweise:
        * Besitzt die Datei mehrere passende Kandidaten, kann ein eigentlich
          relevanter früherer Stream überschrieben werden (Verbesserung möglich).
        * Keine Validierung der Sampling-Rate oder Datenkonsistenz eingebaut.
    """
    eeg_stream, marker_stream = None, None

    for st in streams:
        info = st.get("info", {})
        stype = str(_safe_get(info, "type", "")).lower()
        sname = str(_safe_get(info, "name", "")).lower()
        ch_n = int(float(_safe_get(info, "channel_count", "0")))

        if ("eeg" in stype or "unicorn" in sname) and ch_n >= 1:
            eeg_stream = st
        if "marker" in stype or "markers" in sname:
            marker_stream = st

    return eeg_stream, marker_stream


def eeg_stream_to_raw(eeg_stream: dict, config: DataLoadingConfig) -> mne.io.Raw:
    """Konvertiere einen EEG-Stream in ein `mne.io.Raw` Objekt.

    Schritte:
        1. Sampling-Rate und Rohdaten extrahieren.
        2. Optionale Heuristik-Skalierung (µV → V) basierend auf Medianbetrag.
        3. Generische Kanalnamen erzeugen (Platzhalter, da XDF oft keine eindeutigen Labels liefert).
        4. Kanal-Subset auswählen.
        5. Montage anwenden (soft-fail).
    """
    info = eeg_stream["info"]
    fs = float(_safe_get(info, "nominal_srate", "0"))
    data = np.array(
        eeg_stream["time_series"], dtype=float
    ).T  # shape: (n_channels, n_samples)

    # (Heuristik) Skaliere zu Volt, falls Amplituden eher in µV range liegen.
    if config.auto_scale_to_volts:
        med_abs = float(np.nanmedian(np.abs(data)))
        if med_abs > 1e-3:
            print(f"[INFO] Skaliere von µV zu V (median={med_abs:.1f})")
            data *= 1e-6

    # Generische Kanalnamen – TODO: Echte Namen aus Metadaten extrahieren, falls vorhanden.
    ch_names = [f"EEG{i + 1}" for i in range(data.shape[0])]

    # Erstelle MNE Raw Objekt
    raw = mne.io.RawArray(data, mne.create_info(ch_names, fs, ch_types="eeg"))

    # Kanalfilterung
    if config.channels_keep:
        keep = [ch for ch in config.channels_keep if ch in raw.ch_names]
    else:
        keep = raw.ch_names[: config.max_channels]
    raw.pick_channels(keep)

    # Montage anwenden (weiche Fehlertoleranz)
    if config.montage:
        try:
            raw.set_montage(config.montage, on_missing="ignore")
        except Exception as e:  # pragma: no cover - GUI/IO bedingt
            print(f"[WARN] Montage fehlgeschlagen: {e}")

    return raw


def get_session_paths(
    experiment_sessions: List[Path],
) -> Tuple[Optional[Path], Optional[Path]]:
    """Teile gefundene XDF-Dateien den erwarteten Session-Codes zu.

    Aktuelle Heuristik:
        * Sucht im Ordnerbaum zwei Ebenen über der Datei (`parent.parent.name`).
    Grenzen:
        * Bricht bei abweichender Struktur leicht.
        * Keine Mehrfachzuordnung / Priorisierung.
    """
    sess01_path: Optional[Path] = None
    sess02_path: Optional[Path] = None

    for session in experiment_sessions:
        if session.parent.parent.name == "ses-S001":
            sess01_path = session
        elif session.parent.parent.name == "ses-S002":
            sess02_path = session
        else:
            print(f"Unbekannter Pfad: {session}")

    return sess01_path, sess02_path


def load_session_data(
    session_path: Optional[Path],
    config: DataLoadingConfig,
) -> Tuple[Optional[mne.io.Raw], Optional[pd.DataFrame]]:
    """Lade eine einzelne Session (EEG + Marker-CSV).

    Parameter
    ---------
    session_path:
        Pfad zur XDF-Datei der Session oder `None` (führt zu leerem Ergebnis).
    config:
        Instanz der Lade-Konfiguration.

    Rückgabe
    --------
    raw:
        MNE Raw Objekt oder `None` falls Laden/Parsing scheitert.
    markers:
        DataFrame mit Marker-Einträgen oder `None` wenn keine CSV gefunden.
    """
    if session_path is None:
        return None, None

    print(f"Load Session Data: {session_path}")

    streams, _header = load_xdf_safe(session_path)
    if not streams:
        return None, None

    # Stream-Selektion (EEG + Marker)
    eeg_stream, marker_stream = pick_streams(streams)
    if eeg_stream is None:
        print("[WARN] Kein EEG-Stream gefunden")
        return None, None

    raw = eeg_stream_to_raw(eeg_stream, config)

    # wandel marker_stream in DataFrame um (falls vorhanden)
    markers = None
    if marker_stream and "time_series" in marker_stream:
        marker_data = np.array(marker_stream["time_series"])
        if marker_data.ndim == 1:
            marker_data = marker_data[:, np.newaxis]
        timestamps = np.array(marker_stream["time_stamps"])
        markers = pd.DataFrame(
            marker_data, columns=[f"Marker{i+1}" for i in range(marker_data.shape[1])]
        )
        markers.insert(0, "Timestamp", timestamps)

    if markers is not None:
        print(markers)

    return raw, markers


def load_single_session(
    experiment_dir: Path, config: DataLoadingConfig = None
) -> SessionData:
    """Lade (bis zu) zwei Sessions für einen Teilnehmerordner.

    Annahmen:
        * Maximal zwei XDF-Dateien (assertion sichert das ab – wirft bei Abweichung AssertionError).
        * Teilnehmername = letztes Fragment des Ordnernamens nach '_' Split.
    """
    if config is None:
        config = DataLoadingConfig()

    participant_name = experiment_dir.name.split("_")[-1]
    experiment_sessions = list(experiment_dir.rglob("*.xdf"))
    assert (
        len(experiment_sessions) <= 2
    ), "Mehr als zwei Sessions gefunden – Anpassung nötig"

    indoor_path, outdoor_path = get_session_paths(experiment_sessions)

    indoor_session, indoor_markers = load_session_data(indoor_path, config)
    outdoor_session, outdoor_markers = load_session_data(outdoor_path, config)

    return SessionData(
        participant_name=participant_name,
        indoor_session=indoor_session,
        indoor_markers=indoor_markers,
        outdoor_session=outdoor_session,
        outdoor_markers=outdoor_markers,
    )


def load_all_sessions(
    data_dir: Path, config: DataLoadingConfig = None
) -> List[SessionData]:
    """Lade alle Teilnehmerverzeichnisse innerhalb eines Wurzelordners.

    Fehler pro Teilnehmer führen nicht zum Abbruch (best-effort Aggregation).
    """
    experiment_dirs = [p for p in data_dir.iterdir() if p.is_dir()]
    sessions: List[SessionData] = []

    for experiment_dir in experiment_dirs:
        try:
            session_data = load_single_session(experiment_dir, config)
            sessions.append(session_data)
            print(f"✓ Session geladen: {session_data.participant_name}")
        except Exception as e:  # pragma: no cover
            print(f"✗ Fehler beim Laden {experiment_dir.name}: {e}")

    return sessions


if __name__ == "__main__":
    # Beispiel-Nutzung (manuelles Ausführen des Moduls)
    data_dir = Path("data")
    config = DataLoadingConfig(max_channels=8)

    if data_dir.exists():
        sessions = load_all_sessions(data_dir, config)
        print(f"Insgesamt {len(sessions)} Sessions geladen")
    else:
        print("Data-Verzeichnis nicht gefunden")
