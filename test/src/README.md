# Mini BCI / EEG Test-Projektstruktur

Diese Struktur dient als Ausgangspunkt, um EEG-Daten (z. B. aus MNE‑Beispieldatensätzen) vorzubereiten, Features zu extrahieren und einfache Klassifikations- oder Deep‑Learning‑Modelle zu trainieren. Der Code ist bewusst modular aufgebaut und reich kommentiert – ideal zum Experimentieren vor einem größeren Projekt.

## Ordnerübersicht

```
test/src/
  config/            Zentrale Konfiguration (Pfad- & Pipeline-Parameter)
  data/              (Platzhalter) Roh- & Zwischen-Daten / exportierte Features
  preprocessing/     Schritte zur Signalvorverarbeitung (Filter, ICA-Hooks, Artefakte)
  features/          Feature-Berechnung (Bandpower, CSP, Utility-Funktionen)
  models/            Klassische ML-Pipelines & Deep-Learning-Modelle (EEGNet)
  evaluation/        Metriken & Visualisierung (z. B. Konfusionsmatrix, PSD, Topos)
  utils/             Hilfsfunktionen (Logging, Pfade, Seeds)
  run_example.py     End-to-End Minimalbeispiel mit Sample-Datensatz (MNE)
```

## Minimaler Workflow (klassische Pipeline – CSP + LDA)

1. Rohdaten laden (`mne.datasets.sample` oder eigener Datensatz)
2. Preprocessing: Auswahl EEG-Kanäle, Resampling, Bandpass, Notch, optionale Bad-Channel Interpolation
3. Event-Extraktion & Epoching
4. Feature: CSP (räumliche Filter) + Log-Varianz
5. Klassifikation via LDA oder Logistic Regression (Cross-Validation)
6. Auswertung (Accuracy, ggf. Konfusionsmatrix, Visualisierung)

## Deep Learning (EEGNet – stark vereinfacht)

EEG-Segmente (Kanäle × Samples) werden als Tensoren an ein kleines 1D-CNN (EEGNet-Variante) gegeben. Für echte Projekte: BatchNorm, reguläre Augmentierungen, Balancing & Learning-Rate-Scheduling ergänzen.

## Datensätze

Für lokale Experimente kann der MNE Sample-Datensatz genutzt werden. Für Motor Imagery oder Schlafstadien: BNCI Horizon, BCI Competition, PhysioNet Sleep.

## Erweiterungs-Ideen

- Riemannsche Features (pyriemann) in `features/riemann.py`
- Subjektübergreifende Cross-Validation (Leave-One-Subject-Out)
- Reports (mne.Report) für Preprocessing-Dokumentation
- TorchEEG Integration für standardisierte Datensätze & Modelle
- MLflow oder Weights & Biases für Experiment-Tracking

## Schnellstart

```
cd test/src
python run_example.py --mode classical --limit-epochs 200
python run_example.py --mode deep --limit-epochs 50 --dl-epochs 3
```

Parameter helfen, die Laufzeit gering zu halten (Subsampling von Epochen, niedrige Epoch-Zahl fürs DL-Beispiel).

## Sicherheit & Datenschutz

Bei echten EEG-Datensätzen: Pseudonymisierung, Entfernen direkter Personenbezüge, DSGVO-konforme Speicherung.

---

Viel Erfolg beim Ausprobieren! Ergänze schrittweise tiefergehende Preprocessing- und Evaluationskomponenten.
