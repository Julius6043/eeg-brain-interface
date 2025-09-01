"""Deep learning baseline models for EEG workload decoding.

This module implements an optional baseline based on EEGNet, a compact
convolutional neural network designed specifically for EEG signal
classification【74903913843087†L230-L241】. EEGNet uses depthwise and
separable convolutions to learn spatial and temporal filters with few
parameters, making it suitable for small training datasets and
low‑channel counts. The code below provides functions to build the
EEGNet architecture using TensorFlow/Keras and to train it with
subject‑wise cross‑validation.

Note that deep learning libraries are not installed by default in
this environment. To run the functions in this module the user must
install TensorFlow (>=2.4) or Keras separately.
"""

from __future__ import annotations

from typing import Tuple, Any, Dict

import numpy as np

try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization,
                                         DepthwiseConv2D, SeparableConv2D, AveragePooling2D,
                                         Dropout, Flatten, Dense)
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    tf = None  # type: ignore
    Input = None  # type: ignore
    Conv2D = None  # type: ignore
    BatchNormalization = None  # type: ignore
    DepthwiseConv2D = None  # type: ignore
    SeparableConv2D = None  # type: ignore
    AveragePooling2D = None  # type: ignore
    Dropout = None  # type: ignore
    Flatten = None  # type: ignore
    Dense = None  # type: ignore
    Model = None  # type: ignore
    l2 = None  # type: ignore
    to_categorical = None  # type: ignore
    EarlyStopping = None  # type: ignore

try:
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
except ImportError:
    GroupKFold = None  # type: ignore
    LeaveOneGroupOut = None  # type: ignore
    balanced_accuracy_score = None  # type: ignore
    f1_score = None  # type: ignore
    roc_auc_score = None  # type: ignore

from .config import EEGNetConfig, TrainingConfig


def build_eegnet_model(
    input_shape: Tuple[int, int, int],
    n_classes: int,
    F1: int = 8,
    D: int = 2,
    F2: int = 16,
    kern_length: int = 64,
    dropout_rate: float = 0.5,
    dropout_type: str = 'Dropout',
    reg: float = 0.25,
) -> 'Model':
    """Construct the EEGNet v2.0 architecture.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input data (channels, samples, 1). The final
        dimension is a singleton to allow 2D convolutions.
    n_classes : int
        Number of output classes.
    F1, D, F2 : int
        Hyper‑parameters controlling the number of spatial and depth
        filters. Defaults follow the original EEGNet paper【74903913843087†L230-L241】.
    kern_length : int
        Length of the temporal convolution kernel.
    dropout_rate : float
        Dropout rate applied after convolutional blocks.
    dropout_type : str
        Either 'Dropout' or 'SpatialDropout2D'. Spatial dropout
        randomly drops entire feature maps and may improve
        generalisation.
    reg : float
        L2 regularisation strength.

    Returns
    -------
    tf.keras.Model
        The compiled EEGNet model.
    """
    if tf is None:
        raise ImportError(
            "TensorFlow is required to build EEGNet. Please install tensorflow>=2.4."
        )
    if dropout_type == 'SpatialDropout2D':
        from tensorflow.keras.layers import SpatialDropout2D  # type: ignore
        DropoutLayer = SpatialDropout2D
    else:
        DropoutLayer = Dropout

    input_tensor = Input(shape=input_shape)
    # First temporal convolution
    x = Conv2D(
        filters=F1,
        kernel_size=(1, kern_length),
        padding='same',
        input_shape=input_shape,
        use_bias=False,
        kernel_regularizer=l2(reg),
    )(input_tensor)
    x = BatchNormalization()(x)
    # Depthwise convolution (spatial)
    x = DepthwiseConv2D(
        kernel_size=(input_shape[0], 1),
        depth_multiplier=D,
        use_bias=False,
        depthwise_regularizer=l2(reg),
        padding='valid',
    )(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)
    x = DropoutLayer(dropout_rate)(x)
    # Separable convolution
    x = SeparableConv2D(
        filters=F2,
        kernel_size=(1, 16),
        use_bias=False,
        padding='same',
        depthwise_regularizer=l2(reg),
        pointwise_regularizer=l2(reg),
    )(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('elu')(x)
    x = AveragePooling2D((1, 8))(x)
    x = DropoutLayer(dropout_rate)(x)
    x = Flatten()(x)
    x = Dense(n_classes, activation='softmax', kernel_regularizer=l2(reg))(x)
    model = Model(inputs=input_tensor, outputs=x)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def train_eegnet_crossval(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    eegnet_config: EEGNetConfig,
    training_config: TrainingConfig,
    n_epochs: int = 100,
    batch_size: int = 32,
    patience: int = 10,
) -> Dict[str, Any]:
    """Train EEGNet using subject‑wise cross‑validation.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_channels, n_samples_time)
        Input signals. If eegnet_config.input_length is None the
        temporal dimension of ``X`` is used.
    y : ndarray, shape (n_samples,)
        Integer labels.
    groups : ndarray, shape (n_samples,)
        Subject identifiers for cross‑validation.
    eegnet_config : EEGNetConfig
        Configuration specifying sampling rate, number of channels and
        input length.
    training_config : TrainingConfig
        Cross‑validation strategy and parallelism. Only
        'leave-one-subject-out' and 'group-k-fold' are supported.
    n_epochs : int
        Maximum number of training epochs.
    batch_size : int
        Batch size for model training.
    patience : int
        Early stopping patience on the validation loss.

    Returns
    -------
    dict
        Per‑fold and aggregated performance metrics (balanced accuracy,
        macro F1 and ROC AUC).
    """
    if tf is None:
        raise ImportError(
            "TensorFlow is required to train EEGNet. Please install tensorflow>=2.4."
        )
    # Remove samples with invalid labels
    valid_mask = y >= 0
    X = X[valid_mask]
    y = y[valid_mask]
    groups = groups[valid_mask]

    # Determine input shape
    channels = X.shape[1]
    time_length = X.shape[2] if eegnet_config.input_length is None else int(eegnet_config.input_length)
    # Ensure fixed length by padding/truncating
    if time_length != X.shape[2]:
        if X.shape[2] > time_length:
            X = X[:, :, :time_length]
        else:
            pad_width = time_length - X.shape[2]
            X = np.pad(X, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
    # Reshape to 4D tensor (samples, channels, time, 1)
    X_input = X[:, :, :, np.newaxis].astype(np.float32)
    n_classes = int(np.max(y) + 1)

    # Cross-validation iterator
    cv = None
    if training_config.cv_strategy == 'leave-one-subject-out':
        cv = LeaveOneGroupOut()
    elif training_config.cv_strategy.startswith('group-k-fold'):
        parts = training_config.cv_strategy.split('-')
        n_splits = 5
        if len(parts) == 4:
            try:
                n_splits = int(parts[-1])
            except Exception:
                pass
        cv = GroupKFold(n_splits=n_splits)
    else:
        raise ValueError(f"Unsupported cv_strategy: {training_config.cv_strategy}")

    folds = []
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X_input, y, groups)):
        X_train, X_test = X_input[train_idx], X_input[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # Convert labels to categorical
        Y_train = to_categorical(y_train, num_classes=n_classes)
        Y_test = to_categorical(y_test, num_classes=n_classes)
        # Build a new EEGNet model for each fold to avoid weight leakage
        model = build_eegnet_model((channels, time_length, 1), n_classes)
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)]
        # Fit model
        model.fit(
            X_train, Y_train,
            validation_data=(X_test, Y_test),
            epochs=n_epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=callbacks,
        )
        # Evaluate
        proba = model.predict(X_test)
        y_pred = np.argmax(proba, axis=1)
        ba = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        # Convert probabilities for roc_auc_score: shape (n_samples, n_classes)
        roc_auc = roc_auc_score(y_test, proba, multi_class='ovr', average='macro')
        folds.append({
            'fold': fold_i,
            'balanced_accuracy': ba,
            'f1_macro': f1,
            'roc_auc_macro': roc_auc,
        })
    mean_scores = {
        'balanced_accuracy': float(np.mean([f['balanced_accuracy'] for f in folds])),
        'f1_macro': float(np.mean([f['f1_macro'] for f in folds])),
        'roc_auc_macro': float(np.mean([f['roc_auc_macro'] for f in folds])),
    }
    return {
        'folds': folds,
        'mean': mean_scores,
    }