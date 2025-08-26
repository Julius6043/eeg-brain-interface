import mne
from pathlib import Path
from mne.datasets import sample

p = Path(sample.data_path(verbose=False)) / "MEG" / "sample" / "sample_audvis_raw.fif"
raw = mne.io.read_raw_fif(str(p), preload=False, verbose=False)
print(sorted(set(a["description"] for a in raw.annotations)))
