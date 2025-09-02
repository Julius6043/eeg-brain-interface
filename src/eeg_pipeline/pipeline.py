from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .plot import PlotConfig
from .data_loading import DataLoadingConfig, SessionData, load_all_sessions
from .preprocessing import PreprocessingConfig, preprocess_raw


@dataclass
class PipelineConfig:
    data_loading: DataLoadingConfig
    preprocessing: PreprocessingConfig
    plot: PlotConfig
    output_dir: Optional[Path] = None


class EEGPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.sessions: List[SessionData] = []

    def run(self, data_dir: Path) -> List[SessionData]:
        print("Start EEG-Pipeline...")

        print("\nStep 1: Load data")
        self.sessions = load_all_sessions(data_dir.resolve(), self.config.data_loading)
        print(f"✓ {len(self.sessions)} Sessions loaded")

        print("\nStep 2: Preprocessing")
        for session in self.sessions:
            if session.indoor_session:
                session.indoor_session = preprocess_raw(
                    session.indoor_session,
                    self.config.preprocessing
                )
            if session.outdoor_session:
                session.outdoor_session = preprocess_raw(
                    session.outdoor_session,
                    self.config.preprocessing
                )
        print("✓ Preprocessing abgeschlossen")

        if self.config.plot:
            print(f"\nPlotting step started")
            raw.compute_psd(fmax=50).plot(picks="data", exclude="bads", amplitude=False)
            raw.plot(duration=5, n_channels=30)

        if self.config.output_dir:
            print("\nStep 3: Save processed data")
            self._save_processed_data()

        print("\nPipeline finished!")
        return self.sessions

    def _save_processed_data(self):
        """Speichere verarbeitete Daten."""
        self.config.output_dir.mkdir(exist_ok=True)

        for session in self.sessions:
            session_dir = self.config.output_dir / session.participant_name
            session_dir.mkdir(exist_ok=True)

            if session.indoor_session:
                indoor_path = session_dir / "indoor_processed.fif"
                session.indoor_session.save(str(indoor_path), overwrite=True)

            if session.outdoor_session:
                outdoor_path = session_dir / "outdoor_processed.fif"
                session.outdoor_session.save(str(outdoor_path), overwrite=True)

        print(f"✓ Daten gespeichert in {self.config.output_dir}")


def create_default_config() -> PipelineConfig:
    return PipelineConfig(
        data_loading=DataLoadingConfig(
            max_channels=8,
            montage="standard_1020"
        ),
        preprocessing=PreprocessingConfig(
            l_freq=1.0,
            h_freq=40.0,
            notch_freq=50.0
        )
    )


def main():
    data_dir = Path("../data")
    config = create_default_config()
    config.output_dir = Path("../results/processed")

    pipeline = EEGPipeline(config)
    sessions = pipeline.run(data_dir)

    print(f"\nPipeline abgeschlossen mit {len(sessions)} Sessions")


if __name__ == '__main__':
    main()
