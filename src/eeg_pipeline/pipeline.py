"""High-level EEG Processing Pipeline.

Dieses Modul orchestriert die Hauptschritte der EEG-Verarbeitung:
    1. Laden aller Sessions / Teilnehmer.
    2. Preprocessing (Filter, Referenz, optional ICA).
    3. (Optional) Plot/Visualisierung (Platzhalter).
    4. Speichern der verarbeiteten Daten.

Die Pipeline kapselt damit *Ablauflogik*, ohne die Details der einzelnen
Verarbeitungsschritte (Delegation an Submodule: `data_loading`, `preprocessing`).

Erweiterungsideen:
    * Validierungsschritt integrieren (z.B. mit `validation` Modul) – vor/nach Preprocessing.
    * Event- / Marker-Handling vereinheitlichen und in Raw-Objekte übernehmen.
    * Logging über `logging` statt `print` (für produktive Nutzung / Batch).
    * Konfigurations-Serialisierung (JSON/YAML) für reproduzierbare Runs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from eeg_pipeline.plot import PlotConfig
from eeg_pipeline.data_loading import DataLoadingConfig, SessionData, load_all_sessions
from eeg_pipeline.preprocessing import PreprocessingConfig, preprocess_raw
from eeg_pipeline.marker_annotation import annotate_raw_with_markers
import mne


@dataclass
class PipelineConfig:
    """Aggregierte Konfiguration für gesamten Pipeline-Lauf.

    Attribute
    ---------
    data_loading:
        Parameter für Einlesen / Stream-Selektion.
    preprocessing:
        Parameter für Filterung / ICA etc.
    plot:
        (Optional) Visualisierungs-Parameter (derzeit leerer Platzhalter `PlotConfig`).
    output_dir:
        Zielverzeichnis für persistierte FIF-Dateien; wenn `None`, werden keine Daten gespeichert.
    """

    data_loading: DataLoadingConfig
    preprocessing: PreprocessingConfig
    plot: PlotConfig
    output_dir: Optional[Path] = None


class EEGPipeline:
    """Abstraktion eines gesamten EEG-Verarbeitungsablaufs.

    Die Klasse hält internen Zustand (`sessions`) und erlaubt (später mögliche)
    Erweiterungen wie erneutes Ausführen nur einzelner Stufen.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.sessions: List[SessionData] = []

    def run(self, data_dir: Path) -> List[SessionData]:
        """Starte Pipeline-Ende-zu-Ende.

        Parameter
        ---------
        data_dir:
            Wurzelverzeichnis der Rohdaten (Teilnehmer-Unterordner erwartet).
        """
        print("Start EEG-Pipeline...")

        # 1. Laden
        print("\nStep 1: Load data")
        self.sessions = load_all_sessions(data_dir.resolve(), self.config.data_loading)
        print(f"✓ {len(self.sessions)} Sessions loaded")

        # 2. Preprocessing
        print("\nStep 2: Preprocessing")
        for session in self.sessions:
            if session.indoor_session:
                session.indoor_session = preprocess_raw(
                    session.indoor_session, self.config.preprocessing
                )
            if session.outdoor_session:
                session.outdoor_session = preprocess_raw(
                    session.outdoor_session, self.config.preprocessing
                )
        print("✓ Preprocessing abgeschlossen")

        # 3. Marker-Annotation
        print("\nStep 3: Adding marker-based annotations")
        for session in self.sessions:
            if session.indoor_session and session.indoor_markers is not None:
                session.indoor_session = annotate_raw_with_markers(
                    session.indoor_session, session.indoor_markers
                )
                print(f"  ✓ Indoor annotations added for {session.participant_name}")
            if session.outdoor_session and session.outdoor_markers is not None:
                session.outdoor_session = annotate_raw_with_markers(
                    session.outdoor_session, session.outdoor_markers
                )
                print(f"  ✓ Outdoor annotations added for {session.participant_name}")
        print("✓ Marker-Annotation abgeschlossen")
        """
        # 4. Epoching
        if self.config.epoching:
            print("\nStep 4: Epoching")
            for session in self.sessions:
                if session.indoor_session and session.indoor_session.annotations:
                    session.indoor_session = epoch_raw(session.indoor_session)
                if session.outdoor_session and session.outdoor_session.annotations:
                    session.outdoor_session = epoch_raw(session.outdoor_session)
        print("✓ Epoching abgeschlossen")
        """
        # 5. Plot (Platzhalter) – könnte später differenziert werden (PSD, ERP, …)
        if self.config.plot:
            print("\nPlotting step started (derzeit Platzhalter)")
            # TODO: Implement plotting utilities.
            pass

        # 6. Persistierung
        if self.config.output_dir:
            print("\nStep 4: Save processed data")
            self._save_processed_data()

        print("\nPipeline finished!")
        return self.sessions

    def _save_processed_data(self):
        """Persistiere alle verarbeiteten Sessions als FIF.

        Aktuell: Einfache Ordnerstruktur pro Teilnehmer. Keine Versionierung /
        Metadaten-JSON. Erweiterbar um: Hashing, Parameterprotokoll, QC-Kennzahlen.
        """
        assert self.config.output_dir is not None, "output_dir darf nicht None sein"
        self.config.output_dir.mkdir(exist_ok=True)

        for session in self.sessions:
            session_dir = self.config.output_dir / session.participant_name
            session_dir.mkdir(exist_ok=True)

            print(f"  Speichere {session.participant_name}:")

            if session.indoor_session:
                indoor_path = session_dir / "indoor_processed_raw.fif"
                session.indoor_session.save(str(indoor_path), overwrite=True)
                print("    ✓ Indoor-Session gespeichert")
            else:
                print("    ⚠ Keine Indoor-Session verfügbar")

            if session.outdoor_session:
                outdoor_path = session_dir / "outdoor_processed_raw.fif"
                session.outdoor_session.save(str(outdoor_path), overwrite=True)
                print("    ✓ Outdoor-Session gespeichert")
            else:
                print("    ⚠ Keine Outdoor-Session verfügbar")

        print(f"✓ Daten gespeichert in {self.config.output_dir}")


def create_default_config() -> PipelineConfig:
    """Erzeuge eine einfache Standardkonfiguration.

    Hinweis: `plot=None` deaktiviert den Plot-Schritt.
    """
    return PipelineConfig(
        data_loading=DataLoadingConfig(max_channels=8, montage="standard_1020"),
        plot=None,
        preprocessing=PreprocessingConfig(l_freq=1.0, h_freq=40.0, notch_freq=50.0),
    )


def main():
    """CLI-Einstieg (einfaches Beispiel ohne Argumentparser)."""
    data_dir = Path("data")
    config = create_default_config()
    config.output_dir = Path("results/processed")

    pipeline = EEGPipeline(config)
    sessions = pipeline.run(data_dir)

    print(f"\nPipeline abgeschlossen mit {len(sessions)} Sessions")


if __name__ == "__main__":
    main()
