from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import mne
from mne.io import Raw, read_raw_fif
from pandas.io.stata import invalid_name_doc


@dataclass
class FifExperiment:
    participant_name: str
    indoor_session: Optional[Raw] = None
    outdoor_session: Optional[Raw] = None


def main():
    root_dir = Path(__file__).parent / "results" / "processed"
    experiments = [x for x in root_dir.iterdir() if x.is_dir()]

    fif_experiments = {}

    for experiment in experiments:
        participant_name = experiment.name
        if participant_name != "Lotta":
            continue
        indoor_session_path = experiment / "indoor_processed_raw.fif"
        outdoor_session_path = experiment / "outdoor_processed_raw.fif"

        indoor_session, outdoor_session = None, None

        if indoor_session_path.is_file() and indoor_session_path.exists():
            indoor_session = read_raw_fif(indoor_session_path)

        if outdoor_session_path.is_file() and outdoor_session_path.exists():
            outdoor_session = read_raw_fif(outdoor_session_path)

        fif_experiment = FifExperiment(participant_name, indoor_session, outdoor_session)

        fif_experiments[fif_experiment.participant_name] = fif_experiment

        events, event_id_dict = mne.events_from_annotations(outdoor_session)


if __name__ == '__main__':
    main()
