# pip install mne pyriemann scikit-learn
from mne import Epochs, events_from_annotations
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_score
import numpy as np, mne

raw = mne.io.read_raw_fif("nback_raw.fif", preload=True)
raw.filter(1, 40).notch_filter(50).set_montage("standard_1020")

# Fenster (2 s, 50% overlap) + Labels y + Gruppen=Subjekte
epochs = mne.make_fixed_length_epochs(raw, duration=2.0, overlap=1.0, preload=True)
X = epochs.get_data()  # (n, ch, t)
y = epochs.metadata["workload"].to_numpy()  # z.B. {0..3} oder low/med/high
groups = epochs.metadata["subject"].to_numpy()

pipe = Pipeline(
    [
        ("cov", Covariances(estimator="oas")),
        ("ts", TangentSpace(metric="riemann")),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ]
)
cv = GroupKFold(n_splits=5)
print(
    "bal-acc:",
    cross_val_score(
        pipe, X, y, groups=groups, scoring="balanced_accuracy", cv=cv
    ).mean(),
)
