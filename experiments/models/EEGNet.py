# pip install braindecode torch mne
import torch, numpy as np, mne
from braindecode.models import EEGNetv4, EEGConformer
from braindecode import EEGClassifier
from braindecode.datautil.windowers import create_fixed_length_windows

raw = mne.io.read_raw_fif("nback_raw.fif", preload=True)
raw.filter(1, 40).notch_filter(50).set_montage("standard_1020")
windows = create_fixed_length_windows(
    raw,
    window_size_samples=int(raw.info["sfreq"] * 2),
    window_stride_samples=int(raw.info["sfreq"] * 1),
    preload=True,
)
X = windows.get_data().astype("float32")  # (n, ch, t)
y = windows.metadata["workload"].to_numpy().astype("int64")
n_ch, n_t, n_cls = X.shape[1], X.shape[2], len(np.unique(y))

model = EEGNetv4(
    n_chans=n_ch, n_outputs=n_cls, input_window_samples=n_t, final_conv_length="auto"
)
# modern: model = EEGConformer(n_outputs=n_cls, chs_info=raw.info["chs"],
#                              input_window_seconds=2.0, sfreq=raw.info["sfreq"])
clf = EEGClassifier(
    model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,
    lr=1e-3,
    batch_size=64,
    train_split=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
clf.fit(X, y)
