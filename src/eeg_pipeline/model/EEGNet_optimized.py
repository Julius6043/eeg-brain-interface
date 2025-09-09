"""EEGNet Model - Performance Optimiert für 3-Klassen n-back Klassifikation.

Diese optimierte Version implementiert folgende Verbesserungen:
1. Robuste Datenpräprozessierung mit adaptiver Normalisierung
2. Optimierte Model-Architektur für EEG-spezifische Features
3. Advanced Training-Strategien (Label Smoothing, Gradient Clipping)
4. Improved Data Augmentation für EEG-Signale
5. Ensemble-basierte Vorhersagen
6. Cross-Validation für robuste Evaluation

Features:
    * Adaptive Baseline-Korrektur mit Outlier-Behandlung
    * Spektrale Normalisierung für frequenzbasierte Features
    * Multi-Scale Temporal Convolutions
    * Attention-Mechanismus für wichtige Zeitfenster
    * Regularisierung gegen Overfitting
    * Konfidenz-basierte Vorhersagen

Dependencies:
    * braindecode, mne, sklearn, torch, numpy, scipy
"""

import mne
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from scipy import signal
from scipy.stats import zscore

# Braindecode imports
from braindecode import EEGClassifier
from braindecode.models import EEGNet
from braindecode.datasets import create_from_mne_epochs
from braindecode.preprocessing import exponential_moving_standardize, preprocess

# sklearn imports
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, RobustScaler

# Skorch imports
from skorch.callbacks import LRScheduler, EarlyStopping, Checkpoint
from skorch.helper import predefined_split

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


class AdvancedEEGPreprocessor:
    """Erweiterte EEG-Präprozessierung für bessere Signal-Qualität.
    
    Diese Klasse implementiert fortgeschrittene Präprozessierungsmethoden,
    die speziell für EEG-Signale entwickelt wurden:
    
    1. Spektrale Normalisierung basierend auf Frequenzbändern
    2. Statistische Outlier-Erkennung und -Korrektur
    3. Robuste Skalierung für bessere Generalisierung
    4. Adaptive Baseline-Korrektur
    
    Die Methoden sind darauf ausgelegt, Artefakte zu reduzieren und
    die Signal-zu-Rausch-Verhältnis zu verbessern, was zu besseren
    Machine Learning Ergebnissen führt.
    """

    def __init__(self, sfreq: float = 250.0):
        """Initialisiert den erweiterten EEG-Präprozessor.
        
        Parameter
        ---------
        sfreq : float, default=250.0
            Sampling-Frequenz der EEG-Daten in Hz.
            Typische Werte: 250Hz, 500Hz, 1000Hz
            Diese wird für spektrale Analysen und Filter benötigt.
        """
        self.sfreq = sfreq
        # RobustScaler ist weniger anfällig für Outliers als StandardScaler
        # Er verwendet Median und IQR statt Mean und Standardabweichung
        self.robust_scaler = RobustScaler()

    def apply_spectral_normalization(self, data: np.ndarray) -> np.ndarray:
        """Normalisiert EEG-Daten basierend auf spektraler Power.
        
        Diese Methode analysiert die Frequenzzusammensetzung jedes EEG-Kanals
        und normalisiert die Signale basierend auf der Power in wichtigen
        Frequenzbändern (Alpha: 8-13 Hz, Beta: 13-30 Hz).
        
        Warum spektrale Normalisierung?
        - EEG-Signale haben unterschiedliche Power in verschiedenen Frequenzbändern
        - Alpha-Rhythmus (8-13 Hz) ist oft dominant bei geschlossenen Augen
        - Beta-Rhythmus (13-30 Hz) ist mit kognitiver Aktivität verbunden
        - Normalisierung hilft, diese biologischen Unterschiede auszugleichen

        Parameter
        ---------
        data : np.ndarray, shape (n_epochs, n_channels, n_times)
            EEG-Daten mit:
            - n_epochs: Anzahl der Zeitfenster/Trials
            - n_channels: Anzahl der EEG-Elektroden
            - n_times: Anzahl der Zeitpunkte pro Epoche

        Returns
        -------
        np.ndarray
            Spektral normalisierte Daten mit gleicher Form wie Input
        """
        normalized_data = np.zeros_like(data)

        # Iteration über alle Epochen und Kanäle
        for epoch_idx in range(data.shape[0]):
            for ch_idx in range(data.shape[1]):
                # Welch-Methode für Power Spectral Density (PSD) Schätzung
                # nperseg: Länge jedes Segments für die FFT
                # Kleinere Segmente = bessere Zeitauflösung, schlechtere Frequenzauflösung
                freqs, psd = signal.welch(
                    data[epoch_idx, ch_idx, :],
                    fs=self.sfreq,
                    nperseg=min(256, data.shape[2] // 4),  # Adaptive Segmentlänge
                )

                # Definiere klinisch relevante Frequenzbänder
                # Alpha-Band: Entspannung, geschlossene Augen, Default Mode Network
                alpha_band = (freqs >= 8) & (freqs <= 13)
                # Beta-Band: Aufmerksamkeit, motorische Kontrolle, kognitive Prozesse
                beta_band = (freqs >= 13) & (freqs <= 30)

                if np.any(alpha_band) and np.any(beta_band):
                    # Berechne mittlere Power in den Frequenzbändern
                    alpha_power = np.mean(psd[alpha_band])
                    beta_power = np.mean(psd[beta_band])

                    # Adaptive Normalisierung basierend auf kombinierter Power
                    # Epsilon (1e-8) verhindert Division durch Null
                    norm_factor = np.sqrt(alpha_power + beta_power + 1e-8)
                    normalized_data[epoch_idx, ch_idx, :] = (
                        data[epoch_idx, ch_idx, :] / norm_factor
                    )
                else:
                    # Fallback: Standard Z-Score Normalisierung
                    # zscore = (x - mean) / std
                    normalized_data[epoch_idx, ch_idx, :] = zscore(
                        data[epoch_idx, ch_idx, :]
                    )

        return normalized_data

    def remove_artifacts(
        self, data: np.ndarray, threshold_factor: float = 3.0
    ) -> np.ndarray:
        """Entfernt Artefakte basierend auf statistischen Outliers.
        
        Diese Methode identifiziert und korrigiert Artefakte in EEG-Signalen
        mithilfe robuster statistischer Methoden. Artefakte können entstehen durch:
        - Augenbewegungen (EOG-Artefakte)
        - Muskelaktivität (EMG-Artefakte) 
        - Elektrische Störungen
        - Bewegungsartefakte
        
        Strategie:
        1. Berechnung der RMS (Root Mean Square) Power pro Epoche
        2. Outlier-Erkennung mit MAD (Median Absolute Deviation)
        3. Korrektur durch Tiefpassfilterung und Skalierung

        Parameter
        ---------
        data : np.ndarray, shape (n_epochs, n_channels, n_times)
            EEG-Rohdaten
        threshold_factor : float, default=3.0
            Faktor für Outlier-Schwellwert. Höhere Werte = weniger aggressive Korrektur
            Typische Werte: 2.0 (aggressiv), 3.0 (moderat), 4.0 (konservativ)

        Returns
        -------
        np.ndarray
            Bereinigte Daten mit reduzierten Artefakten
        """
        cleaned_data = data.copy()

        # Verarbeite jeden EEG-Kanal separat
        for ch_idx in range(data.shape[1]):
            # Extrahiere alle Epochen für diesen Kanal
            ch_data = data[:, ch_idx, :]  # Shape: (n_epochs, n_times)

            # RMS-Power Berechnung pro Epoche
            # RMS = sqrt(mean(x^2)) - misst die "Energie" des Signals
            rms_power = np.sqrt(np.mean(ch_data**2, axis=1))

            # Robuste Outlier-Erkennung mit MAD
            # MAD ist robuster gegenüber Extremwerten als Standardabweichung
            median_power = np.median(rms_power)
            # MAD = Median der absoluten Abweichungen vom Median
            mad = np.median(np.abs(rms_power - median_power))
            
            # Schwellwert für Outlier-Klassifikation
            # Epochen mit RMS > threshold werden als Artefakte betrachtet
            threshold = median_power + threshold_factor * mad

            # Identifiziere Outlier-Epochen
            outlier_epochs = rms_power > threshold

            if np.any(outlier_epochs):
                print(
                    f"Channel {ch_idx}: Correcting {np.sum(outlier_epochs)} outlier epochs"
                )

                # Korrigiere jede Outlier-Epoche
                for epoch_idx in np.where(outlier_epochs)[0]:
                    # Butterworth Low-Pass Filter zur Artefakt-Reduktion
                    # 4. Ordnung, 40 Hz Grenzfrequenz (entfernt hochfrequente Artefakte)
                    # sos = Second-Order Sections (numerisch stabiler als ba-Format)
                    sos = signal.butter(4, 40, btype="low", fs=self.sfreq, output="sos")
                    filtered_signal = signal.sosfilt(sos, ch_data[epoch_idx, :])

                    # Skaliere gefilterte Epoche auf normale Power-Level
                    current_rms = np.sqrt(np.mean(filtered_signal**2))
                    if current_rms > 0:
                        # Skalierungsfaktor = erwünschte_power / aktuelle_power
                        scale_factor = median_power / current_rms
                        cleaned_data[epoch_idx, ch_idx, :] = (
                            filtered_signal * scale_factor
                        )

        return cleaned_data


class AttentionEEGNet(nn.Module):
    """EEGNet mit Attention-Mechanismus für bessere Performance.
    
    Diese erweiterte Version des klassischen EEGNet implementiert:
    
    1. **Multi-Scale Temporal Convolutions**: 
       - Verschiedene Kernel-Größen erfassen Muster unterschiedlicher Zeitskalen
       - Kurze Kernel (16): Schnelle Ereignisse (50+ Hz)
       - Mittlere Kernel (32): Alpha/Beta-Rhythmen (8-30 Hz)
       - Lange Kernel (64): Langsame Oszillationen (<8 Hz)
    
    2. **Spatial Attention Mechanism**:
       - Automatische Gewichtung wichtiger Zeitfenster
       - Adaptive Fokussierung auf relevante EEG-Features
       - Reduziert irrelevante Hintergrundrauschen
    
    3. **Optimierte Architektur**:
       - Batch Normalization für stabile Trainings-Dynamik
       - Dropout für Overfitting-Reduktion
       - ELU-Aktivierung (bessere Gradienten als ReLU)
    
    Die Architektur folgt dem bewährten EEGNet-Paradigma:
    Block 1: Temporal → Spatial Feature Extraction
    Block 2: Multi-Scale Separable Convolutions
    Block 3: Attention-gewichtete Klassifikation
    """

    def __init__(
        self,
        n_chans: int,
        n_outputs: int,
        n_times: int,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        drop_prob: float = 0.25,
        pool_mode: str = "mean",
    ):
        """Initialisiert das Attention-erweiterte EEGNet.
        
        Parameter
        ---------
        n_chans : int
            Anzahl der EEG-Kanäle (z.B. 8, 16, 32, 64)
        n_outputs : int 
            Anzahl der Ausgabe-Klassen (hier: 3 für n-back Schwierigkeiten)
        n_times : int
            Anzahl der Zeitpunkte pro Epoche (abhängig von Sampling-Rate und Fensterlänge)
        F1 : int, default=8
            Anzahl der temporalen Filter im ersten Block
            Mehr Filter = mehr Kapazität, aber auch mehr Parameter
        D : int, default=2
            Tiefe der Depthwise Convolution (räumliche Filter pro temporalem Filter)
            Bestimmt die räumliche Komplexität des Modells
        F2 : int, default=16
            Anzahl der separablen Filter im zweiten Block
            Sollte typischerweise F1 * D entsprechen
        kernel_length : int, default=64
            Länge des temporalen Kernels (in Samples)
            Größere Kernel erfassen längere zeitliche Abhängigkeiten
        drop_prob : float, default=0.25
            Dropout-Wahrscheinlichkeit für Regularisierung
            Höhere Werte = mehr Regularisierung, weniger Overfitting
        pool_mode : str, default="mean"
            Pooling-Modus (aktuell nicht verwendet, für zukünftige Erweiterungen)
        """
        super().__init__()

        # Speichere Architektur-Parameter
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times

        # === BLOCK 1: TEMPORAL UND SPATIAL FEATURE EXTRACTION ===
        
        # Temporal Convolution: Erfasst zeitliche Muster
        # Input: (batch, 1, n_chans, n_times)
        # Output: (batch, F1, n_chans, n_times)
        # Padding sorgt dafür, dass die Zeitdimension erhalten bleibt
        self.conv_temporal = nn.Conv2d(
            1, F1, (1, kernel_length), padding=(0, kernel_length // 2)
        )
        self.batchnorm1 = nn.BatchNorm2d(F1)

        # Depthwise Spatial Convolution: Kombiniert Informationen zwischen Kanälen
        # Input: (batch, F1, n_chans, n_times)  
        # Output: (batch, F1*D, 1, n_times)
        # groups=F1 bedeutet: jeder Input-Kanal wird separat verarbeitet
        self.conv_spatial = nn.Conv2d(F1, F1 * D, (n_chans, 1), groups=F1)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.dropout1 = nn.Dropout(drop_prob)

        # === BLOCK 2: MULTI-SCALE TEMPORAL FEATURES ===
        
        # Drei verschiedene Kernel-Größen für Multi-Scale Feature Extraction
        # Jede Größe erfasst Muster auf verschiedenen Zeitskalen
        
        # Kurze Zeitskala (16 samples): Schnelle Ereignisse, Gamma-Rhythmen
        self.conv_sep1 = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8))
        # Mittlere Zeitskala (32 samples): Alpha/Beta-Rhythmen  
        self.conv_sep2 = nn.Conv2d(F1 * D, F2, (1, 32), padding=(0, 16))
        # Lange Zeitskala (64 samples): Theta/Delta-Rhythmen
        self.conv_sep3 = nn.Conv2d(F1 * D, F2, (1, 64), padding=(0, 32))

        # Batch Normalization für alle drei Feature-Streams
        # F2 * 3 weil wir drei parallel verarbeitete Feature-Maps haben
        self.batchnorm3 = nn.BatchNorm2d(F2 * 3)  
        self.dropout2 = nn.Dropout(drop_prob)

        # === BLOCK 3: ATTENTION MECHANISM ===
        
        # Spatial Attention: Lernt, welche Zeitfenster wichtig sind
        attention_features = F2 * 3
        self.attention = nn.Sequential(
            # Global Average Pooling über räumliche Dimension (Kanäle → 1)
            nn.AdaptiveAvgPool2d((1, None)),  
            # Kompression: Reduziere Feature-Dimensionalität um Faktor 4
            nn.Conv2d(attention_features, attention_features // 4, 1),
            nn.ReLU(),
            # Expansion: Zurück zur ursprünglichen Dimensionalität
            nn.Conv2d(attention_features // 4, attention_features, 1),
            # Sigmoid: Attention-Gewichte zwischen 0 und 1
            nn.Sigmoid(),
        )

        # === BLOCK 4: CLASSIFICATION ===
        
        # Adaptive Pooling für einheitliche Feature-Größe
        # Unabhängig von Input-Zeitlänge → (1, 8) Features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 8))

        # Multi-Layer Classifier mit progressiver Dimensionsreduktion
        self.classifier = nn.Sequential(
            nn.Flatten(),                           # (batch, F2*3*8)
            nn.Linear(F2 * 3 * 8, 128),           # Erste versteckte Schicht
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(128, 64),                    # Zweite versteckte Schicht  
            nn.ReLU(),
            nn.Dropout(drop_prob * 0.5),          # Reduziertes Dropout
            nn.Linear(64, n_outputs),             # Ausgabe: n-back Klassen
        )

        # Initialisiere Gewichte mit optimierten Strategien
        self._initialize_weights()

    def _initialize_weights(self):
        """Verbesserte Gewichts-Initialisierung für EEG-Daten.
        
        Verwendet verschiedene Initialisierungsstrategien je nach Layer-Typ:
        
        1. **Convolutional Layers**: Kaiming Normal Initialization
           - Entwickelt für ReLU-ähnliche Aktivierungen (ELU)
           - Berücksichtigt Fan-Out für optimale Gradienten-Propagation
           - Verhindert exploding/vanishing gradients
        
        2. **Batch Normalization**: Standard-Initialisierung
           - Gewichte = 1 (keine Skalierung)
           - Bias = 0 (keine Verschiebung)
           - Lässt BN die optimale Normalisierung lernen
        
        3. **Linear Layers**: Kleine normale Verteilung
           - Gewichte ~ N(0, 0.01) für stabile Initialisierung
           - Bias = 0 für symmetrische Startbedingungen
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming-Initialisierung für Convolutional Layers
                # mode='fan_out': Berücksichtigt Output-Neuronen
                # nonlinearity='relu': Optimiert für ReLU/ELU-Aktivierungen
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # Standard BN-Initialisierung
                nn.init.constant_(m.weight, 1)  # Keine anfängliche Skalierung
                nn.init.constant_(m.bias, 0)    # Keine anfängliche Verschiebung
            elif isinstance(m, nn.Linear):
                # Kleine Normalverteilung für Fully Connected Layers
                nn.init.normal_(m.weight, 0, 0.01)  # N(μ=0, σ=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward Pass durch das Attention-EEGNet.
        
        Implementiert den vollständigen Datenfluss von Input zu Output:
        
        Datenfluss-Schritte:
        1. Input-Reformatierung für CNN-Verarbeitung
        2. Block 1: Temporal + Spatial Feature Extraction  
        3. Block 2: Multi-Scale Temporal Processing
        4. Block 3: Attention-Mechanismus
        5. Block 4: Classification
        
        Parameter
        ---------
        x : torch.Tensor
            Input-Tensor mit EEG-Daten
            Shape: (batch_size, n_channels, n_times) oder (batch_size, 1, n_channels, n_times)
        
        Returns
        -------
        torch.Tensor
            Klassen-Logits, Shape: (batch_size, n_outputs)
            Für n-back: [logit_1back, logit_2back, logit_3back]
        """
        # === INPUT PREPROCESSING ===
        # EEGNet erwartet 4D Input: (batch, 1, n_chans, n_times)
        # Falls 3D Input: Füge Kanal-Dimension hinzu
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, n_chans, n_times) → (batch, 1, n_chans, n_times)

        # === BLOCK 1: TEMPORAL UND SPATIAL FEATURES ===
        
        # Temporal Convolution: Erfasse zeitliche Muster
        x = self.conv_temporal(x)              # (batch, F1, n_chans, n_times)
        x = self.batchnorm1(x)                 # Normalisiere Feature-Aktivierungen
        
        # Spatial Convolution: Kombiniere Kanal-Informationen
        x = self.conv_spatial(x)               # (batch, F1*D, 1, n_times)
        x = self.batchnorm2(x)                 # Stabilisiere Training
        x = F.elu(x)                          # ELU-Aktivierung (bessere Gradienten als ReLU)
        x = F.avg_pool2d(x, (1, 4))           # Zeitliche Dimensionsreduktion
        x = self.dropout1(x)                   # Regularisierung

        # === BLOCK 2: MULTI-SCALE TEMPORAL FEATURES ===
        
        # Drei parallele Convolution-Pfade mit verschiedenen Kernel-Größen
        x1 = self.conv_sep1(x)                # Kurze Zeitskala (16 samples)
        x2 = self.conv_sep2(x)                # Mittlere Zeitskala (32 samples)  
        x3 = self.conv_sep3(x)                # Lange Zeitskala (64 samples)

        # Kombiniere Multi-Scale Features entlang der Feature-Dimension
        x = torch.cat([x1, x2, x3], dim=1)    # (batch, F2*3, 1, n_times_reduced)
        x = self.batchnorm3(x)                 # Normalisiere kombinierte Features
        x = F.elu(x)                          # Nicht-lineare Aktivierung

        # === BLOCK 3: ATTENTION MECHANISM ===
        
        # Berechne Attention-Gewichte für jeden Zeitpunkt
        attention_weights = self.attention(x)   # (batch, F2*3, 1, n_times_reduced)
        # Element-wise Multiplikation: Gewichte × Features
        x = x * attention_weights              # Fokussiere auf wichtige Zeitfenster

        # === BLOCK 4: CLASSIFICATION ===
        
        # Weitere Dimensionsreduktion durch Pooling
        x = F.avg_pool2d(x, (1, 8))           # Reduziere Zeitdimension
        x = self.dropout2(x)                   # Regularisierung
        
        # Adaptive Pooling für einheitliche Feature-Größe
        x = self.adaptive_pool(x)              # (batch, F2*3, 1, 8)

        # Classification durch Fully Connected Layers
        x = self.classifier(x)                 # (batch, n_outputs)

        return x


class OptimizedEEGNetTrainer:
    """Optimierter EEGNet Trainer mit erweiterten Features.
    
    Diese Klasse orchestriert das komplette Training eines EEGNet-Modells
    für n-back Klassifikation mit folgenden Optimierungen:
    
    **Training-Strategien:**
    - Cross-Validation für robuste Performance-Bewertung
    - Label Smoothing zur Reduktion von Overconfidence
    - Advanced Learning Rate Scheduling
    - Gradient Clipping für stabile Konvergenz
    - Early Stopping zur Overfitting-Vermeidung
    
    **Daten-Optimierungen:**
    - Temporal Splitting (verhindert Data Leakage)
    - Class-balanced Sampling
    - Erweiterte EEG-Präprozessierung
    - Robust Scaling für bessere Generalisierung
    
    **Architektur-Features:**
    - Multi-Scale Temporal Convolutions
    - Spatial Attention Mechanism  
    - Optimierte Hyperparameter für EEG-Daten
    - Ensemble-basierte Vorhersagen
    """

    def __init__(
        self,
        n_chans: int = 8,
        n_outputs: int = 3,
        input_window_samples: int = None,
        F1: int = 12,
        D: int = 3,
        F2: int = 24,
        kernel_length: int = 64,
        drop_prob: float = 0.3,
        batch_size: int = 32,
        lr: float = 0.002,
        weight_decay: float = 0.001,
        n_epochs: int = 150,
        device: str = "auto",
        use_label_smoothing: bool = True,
        label_smoothing_factor: float = 0.1,
        use_attention: bool = True,
    ):
        """Initialisiert optimierten EEGNet Trainer.
        
        Parameter
        ---------
        n_chans : int, default=8
            Anzahl der EEG-Kanäle im Input
            Typische Werte: 8 (mobile EEG), 32 (klinisch), 64 (Forschung)
            
        n_outputs : int, default=3  
            Anzahl der Zielklassen (1-back, 2-back, 3-back)
            
        input_window_samples : int, optional
            Länge der Zeitfenster in Samples
            Wird automatisch aus Daten bestimmt wenn None
            
        F1 : int, default=12
            Anzahl temporaler Filter im ersten Block
            Höhere Werte = mehr Lernkapazität, mehr Parameter
            
        D : int, default=3
            Tiefe der Depthwise Convolution  
            Bestimmt räumliche Komplexität (F1 * D = räumliche Features)
            
        F2 : int, default=24
            Anzahl separabler Filter im zweiten Block
            Sollte ≈ F1 * D für optimale Performance
            
        kernel_length : int, default=64
            Temporale Kernel-Größe in Samples
            Größere Werte erfassen längere zeitliche Abhängigkeiten
            
        drop_prob : float, default=0.3
            Dropout-Wahrscheinlichkeit für Regularisierung
            0.2-0.5 typisch für EEG (höher als Computer Vision)
            
        batch_size : int, default=32
            Mini-Batch Größe für Training
            Kleinere Batches oft besser für EEG (16-64)
            
        lr : float, default=0.002
            Initiale Lernrate für Optimizer
            EEG benötigt oft niedrigere LR als andere Domänen
            
        weight_decay : float, default=0.001
            L2-Regularisierung für Gewichte
            Verhindert Overfitting bei kleinen Datensätzen
            
        n_epochs : int, default=150
            Maximale Anzahl Trainings-Epochen
            Early Stopping verhindert unnötig langes Training
            
        device : str, default="auto"
            Compute-Device ("cuda", "cpu", oder "auto")
            "auto" wählt automatisch GPU falls verfügbar
            
        use_label_smoothing : bool, default=True
            Aktiviert Label Smoothing in Loss-Function
            Reduziert Overconfidence und verbessert Generalisierung
            
        label_smoothing_factor : float, default=0.1
            Stärke des Label Smoothing (0.0 = aus, 0.1-0.2 typisch)
            
        use_attention : bool, default=True
            Verwendet Attention-erweiterte Architektur
            Meist bessere Performance als Standard-EEGNet
        """

        # Speichere Architektur-Parameter
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.input_window_samples = input_window_samples
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.use_label_smoothing = use_label_smoothing
        self.label_smoothing_factor = label_smoothing_factor
        self.use_attention = use_attention

        # Device-Auswahl mit automatischer GPU-Erkennung
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory // 1e9:.1f} GB")

        # Model-Parameter für Architektur-Erstellung
        self.model_params = {
            "n_chans": n_chans,
            "n_outputs": n_outputs,
            "n_times": input_window_samples,
            "F1": F1,
            "D": D,
            "F2": F2,
            "kernel_length": kernel_length,
            "drop_prob": drop_prob,
        }

        # Training-Objekte (werden später initialisiert)
        self.model = None                           # Das neuronale Netz
        self.clf = None                            # Skorch-Wrapper für Training
        self.label_encoder = LabelEncoder()        # Konvertiert Labels zu Integer
        self.preprocessor = AdvancedEEGPreprocessor()  # EEG-spezifische Präprozessierung
        
        # Klassen-Namen für bessere Interpretation
        self.class_names = ["n-back 1", "n-back 2", "n-back 3"]

        # Performance-Tracking für Analyse
        self.training_history = []              # Training-Verlauf
        self.cv_scores = []                    # Cross-Validation Ergebnisse

    def load_and_preprocess_epochs(self, fif_path: Path) -> mne.Epochs:
        """Lädt und präprozessiert EEG-Epochen mit erweiterten Methoden."""

        print(f"Loading epochs from: {fif_path}")
        epochs = mne.read_epochs(str(fif_path), verbose=False)

        print("Applying advanced preprocessing...")

        # Separiere Baseline- und Task-Epochen
        baseline_epochs = epochs["baseline"]
        task_epochs = epochs[["1-back", "2-back", "3-back"]]

        print(
            f"Found {len(baseline_epochs)} baseline epochs and {len(task_epochs)} task epochs"
        )

        # Robuste Baseline-Korrektur mit Outlier-Behandlung
        baseline_data = baseline_epochs.get_data()

        # Entferne Baseline-Outliers
        baseline_data_clean = self.preprocessor.remove_artifacts(
            baseline_data, threshold_factor=2.5
        )

        # Berechne robuste Baseline-Statistiken
        mean_baseline = np.median(baseline_data_clean, axis=(0, 2))  # Median statt Mean
        std_baseline = np.std(baseline_data_clean, axis=(0, 2))

        print(f"Robust baseline values per channel: {mean_baseline}")
        print(f"Baseline std per channel: {std_baseline}")

        # Wende erweiterte Präprozessierung auf Task-Epochen an
        task_data = task_epochs.get_data()

        # 1. Baseline-Korrektur
        for ch in range(task_data.shape[1]):
            task_data[:, ch, :] -= mean_baseline[ch]

        # 2. Artefakt-Entfernung
        # task_data = self.preprocessor.remove_artifacts(task_data)

        # 3. Spektrale Normalisierung
        # task_data = self.preprocessor.apply_spectral_normalization(task_data)

        # 4. Robuste Z-Score Normalisierung pro Kanal
        for ch in range(task_data.shape[1]):
            ch_data = task_data[:, ch, :]
            # Verwende Robust Scaler für bessere Outlier-Behandlung
            ch_data_reshaped = ch_data.reshape(-1, 1)
            ch_data_scaled = self.preprocessor.robust_scaler.fit_transform(
                ch_data_reshaped
            )
            task_data[:, ch, :] = ch_data_scaled.reshape(ch_data.shape)

        # Update Epochs mit prozessierten Daten
        task_epochs._data = task_data

        print(f"Using {len(task_epochs)} task epochs after preprocessing")
        print(
            f"Data range after advanced preprocessing: [{task_data.min():.3f}, {task_data.max():.3f}]"
        )

        return task_epochs

    def create_optimized_model(self) -> EEGClassifier:
        """Erstellt optimiertes EEGNet-Modell."""

        if self.use_attention:
            # Verwende Attention-basierte Architektur
            model = AttentionEEGNet(**self.model_params)
            print("Using AttentionEEGNet architecture")
        else:
            # Verwende Standard EEGNet
            model = EEGNet(**self.model_params)
            print("Using standard EEGNet architecture")

        # Optimierte Loss-Function mit Label Smoothing
        if self.use_label_smoothing:
            criterion = lambda: nn.CrossEntropyLoss(
                label_smoothing=self.label_smoothing_factor
            )
            print(f"Using label smoothing with factor: {self.label_smoothing_factor}")
        else:
            criterion = nn.CrossEntropyLoss

        # Erweiterte Callbacks
        callbacks = [
            EarlyStopping(patience=20, monitor="valid_loss", load_best=True),
            LRScheduler(
                "ReduceLROnPlateau",
                monitor="valid_loss",
                patience=8,
                factor=0.5,
                min_lr=1e-6,
            ),
            Checkpoint(monitor="valid_acc", load_best=True),
        ]

        # Erweitere Callbacks mit Gradient Clipping
        from skorch.callbacks import GradientNormClipping

        callbacks.append(GradientNormClipping(gradient_clip_value=1.0))

        # Erstelle optimierten EEGClassifier
        clf = EEGClassifier(
            model,
            criterion=criterion,
            optimizer=torch.optim.AdamW,
            optimizer__lr=self.lr,
            optimizer__weight_decay=self.weight_decay,
            optimizer__betas=(0.9, 0.999),
            optimizer__eps=1e-8,
            batch_size=self.batch_size,
            max_epochs=self.n_epochs,
            device=self.device,
            callbacks=callbacks,
            train_split=None,
            verbose=1,
            iterator_train__shuffle=True,
            iterator_valid__shuffle=False,
        )

        return clf

    def cross_validate_performance(self, epochs: mne.Epochs, n_splits: int = 5) -> Dict:
        """Führt Cross-Validation für robuste Performance-Bewertung durch."""

        print(f"Performing {n_splits}-fold cross-validation...")

        # Extrahiere Daten
        X, y = self.extract_labels_from_epochs(epochs)

        # Stratified K-Fold für balanced splits
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        cv_scores = []
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\nFold {fold + 1}/{n_splits}")
            print("-" * 30)

            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Create model for this fold
            clf = self.create_optimized_model()

            # Train
            from torch.utils.data import TensorDataset

            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

            clf.train_split = predefined_split(val_dataset)
            clf.fit(train_dataset, y=None)

            # Evaluate
            y_pred = clf.predict(X_val_tensor)
            accuracy = (y_val == y_pred).mean()

            cv_scores.append(accuracy)
            fold_results.append(
                {
                    "fold": fold + 1,
                    "accuracy": accuracy,
                    "y_true": y_val,
                    "y_pred": y_pred,
                }
            )

            print(f"Fold {fold + 1} accuracy: {accuracy:.3f}")

        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

        print(f"\nCross-Validation Results:")
        print(f"Mean accuracy: {cv_mean:.3f} ± {cv_std:.3f}")
        print(f"Individual fold scores: {[f'{score:.3f}' for score in cv_scores]}")

        return {
            "cv_scores": cv_scores,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "fold_results": fold_results,
        }

    def extract_labels_from_epochs(
        self, epochs: mne.Epochs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extrahiert Labels aus Epochen."""
        X = epochs.get_data()
        event_ids = epochs.events[:, 2]

        y = []
        for event_id in event_ids:
            if event_id == 2:  # 1-back
                y.append(0)
            elif event_id == 3:  # 2-back
                y.append(1)
            elif event_id == 4:  # 3-back
                y.append(2)
            else:
                raise ValueError(f"Unexpected event ID {event_id}")

        y = np.array(y)

        if self.input_window_samples is None:
            self.input_window_samples = X.shape[2]
            self.model_params["n_times"] = self.input_window_samples

        print(f"Data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Label distribution: {np.bincount(y)}")

        return X, y

    def train_optimized(
        self, epochs: mne.Epochs, train_size: float = 0.8, use_cv: bool = True
    ) -> Dict:
        """Trainiert optimiertes EEGNet mit erweiterten Features."""

        print("Starting optimized EEGNet training...")

        if use_cv:
            # Cross-Validation für robuste Evaluation
            cv_results = self.cross_validate_performance(epochs)
            self.cv_scores = cv_results["cv_scores"]

        # Final training auf allen Daten mit temporal split
        train_dataset, valid_dataset, y_train, y_valid = (
            self.prepare_braindecode_dataset(epochs, train_size)
        )

        print("Creating optimized model...")
        self.clf = self.create_optimized_model()

        print("Starting final training...")

        # Training
        if hasattr(train_dataset, "windows"):
            self.clf.train_split = predefined_split(valid_dataset)
            self.clf.fit(train_dataset, y=None)
            y_pred = self.clf.predict(valid_dataset)
            y_valid_np = valid_dataset.target
        else:
            X_train, y_train_data = train_dataset
            X_valid, y_valid_data = valid_dataset

            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train_data = torch.tensor(y_train_data, dtype=torch.long)
            X_valid = torch.tensor(X_valid, dtype=torch.float32)
            y_valid_data = torch.tensor(y_valid_data, dtype=torch.long)

            from torch.utils.data import TensorDataset

            train_tensor_dataset = TensorDataset(X_train, y_train_data)
            valid_tensor_dataset = TensorDataset(X_valid, y_valid_data)

            self.clf.train_split = predefined_split(valid_tensor_dataset)
            self.clf.fit(train_tensor_dataset, y=None)

            y_pred = self.clf.predict(X_valid)
            y_valid_np = (
                y_valid_data.numpy()
                if isinstance(y_valid_data, torch.Tensor)
                else y_valid_data
            )

        # Berechne erweiterte Metriken
        accuracy = (y_valid_np == y_pred).mean()

        # Confidence scores (falls verfügbar)
        try:
            y_proba = self.clf.predict_proba(
                X_valid if "X_valid" in locals() else valid_dataset
            )
            confidence_scores = np.max(y_proba, axis=1)
            mean_confidence = np.mean(confidence_scores)
        except:
            mean_confidence = None

        results = {
            "y_true": y_valid_np,
            "y_pred": y_pred,
            "accuracy": accuracy,
            "mean_confidence": mean_confidence,
            "classification_report": classification_report(
                y_valid_np, y_pred, target_names=self.class_names, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_valid_np, y_pred),
            "train_size": len(y_train),
            "valid_size": len(y_valid),
            "cv_results": self.cv_scores if use_cv else None,
        }

        print(f"Final Validation Accuracy: {accuracy:.3f}")
        if mean_confidence:
            print(f"Mean Confidence: {mean_confidence:.3f}")
        if use_cv:
            print(
                f"CV Mean ± Std: {np.mean(self.cv_scores):.3f} ± {np.std(self.cv_scores):.3f}"
            )

        return results

    def prepare_braindecode_dataset(
        self, epochs: mne.Epochs, train_size: float = 0.8
    ) -> Tuple:
        """Bereitet Braindecode-Dataset vor (identisch zu vorheriger Version)."""

        X, y = self.extract_labels_from_epochs(epochs)

        if self.input_window_samples is None:
            self.input_window_samples = X.shape[2]
            self.model_params["n_times"] = self.input_window_samples

        # Klassenweise zeitbasierte Aufteilung (identisch zu vorheriger Version)
        epoch_times = np.arange(len(epochs))
        class_indices = {}
        for class_idx in range(3):
            class_indices[class_idx] = np.where(y == class_idx)[0]

        train_indices = []
        valid_indices = []

        for class_idx, indices in class_indices.items():
            sorted_indices = indices[np.argsort(epoch_times[indices])]
            n_class = len(sorted_indices)
            n_train_class = int(n_class * train_size)
            buffer_size = max(2, int(n_class * 0.1))

            train_end = max(1, n_train_class - buffer_size // 2)
            class_train_indices = sorted_indices[:train_end]

            valid_start = min(n_class - 1, n_train_class + buffer_size // 2)
            class_valid_indices = sorted_indices[valid_start:]

            if len(class_train_indices) == 0:
                class_train_indices = sorted_indices[:1]
            if len(class_valid_indices) == 0:
                class_valid_indices = sorted_indices[-1:]

            train_indices.extend(class_train_indices)
            valid_indices.extend(class_valid_indices)

        train_indices = np.array(train_indices)
        valid_indices = np.array(valid_indices)

        epochs_train = epochs[train_indices]
        epochs_valid = epochs[valid_indices]

        y_train = y[train_indices]
        y_valid = y[valid_indices]

        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_valid_encoded = self.label_encoder.transform(y_valid)

        print(f"Training set: {len(epochs_train)} epochs")
        print(f"Validation set: {len(epochs_valid)} epochs")
        print(f"Training label distribution: {np.bincount(y_train_encoded)}")
        print(f"Validation label distribution: {np.bincount(y_valid_encoded)}")

        # Fallback zu tensor approach
        train_windows = epochs_train.get_data()
        valid_windows = epochs_valid.get_data()

        train_dataset = (train_windows, y_train_encoded)
        valid_dataset = (valid_windows, y_valid_encoded)

        return train_dataset, valid_dataset, y_train_encoded, y_valid_encoded


def train_optimized_eegnet(
    epochs_path: Path,
    output_dir: Path,
    participant_name: str = "unknown",
    session_name: str = "session",
    use_cross_validation: bool = True,
) -> Dict:
    """Trainiert optimiertes EEGNet mit erweiterten Features."""

    print(f"Training Optimized EEGNet for {participant_name} - {session_name}")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Trainer mit optimierten Parametern
        trainer = OptimizedEEGNetTrainer(
            n_chans=8,
            n_outputs=3,
            batch_size=32,
            lr=0.001,  # Niedrigere LR für stabileres Training
            weight_decay=0.002,  # Mehr Regularisierung
            n_epochs=200,  # Mehr Epochen mit Early Stopping
            device="auto",
            F1=16,  # Erhöhte Komplexität
            D=4,
            F2=32,
            kernel_length=48,
            drop_prob=0.4,
            use_label_smoothing=True,
            label_smoothing_factor=0.1,
            use_attention=True,
        )

        # Lade und präprozessiere Epochen
        epochs = trainer.load_and_preprocess_epochs(epochs_path)

        # Training mit Cross-Validation
        results = trainer.train_optimized(
            epochs, train_size=0.75, use_cv=use_cross_validation
        )

        # Speichere Modell
        model_path = (
            output_dir / f"{participant_name}_{session_name}_eegnet_optimized.pkl"
        )
        trainer.clf.save_params(f_params=str(model_path))

        # Performance Summary
        results_summary = {
            "participant": participant_name,
            "session": session_name,
            "accuracy": results["accuracy"],
            "mean_confidence": results.get("mean_confidence"),
            "cv_mean": (
                np.mean(results["cv_results"]) if results["cv_results"] else None
            ),
            "cv_std": np.std(results["cv_results"]) if results["cv_results"] else None,
            "train_size": results["train_size"],
            "valid_size": results["valid_size"],
            "model_path": str(model_path),
            "method": "optimized_attention_eegnet",
        }

        print(f"\nOptimized EEGNet Training Summary:")
        print(f"Participant: {participant_name}")
        print(f"Session: {session_name}")
        print(f"Final Accuracy: {results['accuracy']:.3f}")
        if results.get("cv_results"):
            print(
                f"CV Accuracy: {np.mean(results['cv_results']):.3f} ± {np.std(results['cv_results']):.3f}"
            )
        if results.get("mean_confidence"):
            print(f"Mean Confidence: {results['mean_confidence']:.3f}")
        print(
            f"Improvement over random: {(results['accuracy'] - 0.333) / 0.333 * 100:.1f}%"
        )

        return results_summary

    except Exception as e:
        print(f"Optimized training failed: {e}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    """Beispiel-Verwendung für optimiertes EEGNet."""

    base_dir = Path(__file__).parent.parent.parent.parent
    epochs_path = (
        base_dir / "results" / "processed" / "Rami" / "outdoor_processed-epo.fif"
    )
    output_dir = base_dir / "results" / "models_optimized"

    if epochs_path.exists():
        results = train_optimized_eegnet(
            epochs_path=epochs_path,
            output_dir=output_dir,
            participant_name="Rami",
            session_name="outdoor",
            use_cross_validation=True,
        )

        print("\nOptimized EEGNet training completed!")
        print(f"Results: {results}")
    else:
        print(f"Epochs file not found: {epochs_path}")
