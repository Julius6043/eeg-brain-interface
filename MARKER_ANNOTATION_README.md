# EEG Pipeline mit Marker-Annotation

## Übersicht

Die EEG-Pipeline wurde erweitert um automatische Marker-basierte Annotation von experimentellen Blöcken. Die Pipeline erkennt jetzt automatisch n-back Blöcke aus den Marker-Streams und annotiert sie entsprechend ihrer Schwierigkeit.

## Neue Funktionalität

### 1. Marker-Annotation Modul (`src/eeg_pipeline/marker_annotation.py`)

- **Automatische n-back Erkennung**: Verwendet das `Block_difficulty_extractor.py` Skript um n-back Level (0, 1, 2, 3) zu bestimmen
- **Block-Extraktion**: Identifiziert experimentelle Blöcke anhand von `main_block_X_start` Markern
- **Baseline-Extraktion**: Erkennt Baseline-Perioden aus `baseline_start` und `baseline_end` Markern
- **Erweiterte Zeitinformationen**: Speichert sowohl Sekunden als auch Sample-Informationen in den Annotationen
- **MNE-Integration**: Erstellt MNE Annotations-Objekte die in Raw-Dateien gespeichert werden

### 2. Erweiterte Pipeline (`src/eeg_pipeline/pipeline.py`)

Die Pipeline wurde um einen neuen Schritt erweitert:

1. **Laden** - XDF-Dateien und Marker-Streams einlesen
2. **Preprocessing** - Filter, Referenz, optional ICA  
3. **🆕 Marker-Annotation** - Block-Annotationen hinzufügen
4. **Plotting** - (optional)
5. **Speichern** - FIF-Dateien mit Annotationen

### 3. Annotation Format

Die Annotationen folgen einem strukturierten Schema nur mit Zeitinformationen in Sekunden:

**Baseline-Annotationen**:
`baseline_{nummer}_onset_{onset_sekunden}s_dur_{dauer_sekunden}s`

**Block-Annotationen**:
`block_{nummer:02d}_nback_{level}_onset_{onset_sekunden}s_dur_{dauer_sekunden}s`

Beispiele:
- `baseline_1_onset_0.0s_dur_121.0s` - Baseline 1, Start bei 0s, 121s Dauer
- `block_00_nback_0_onset_129.5s_dur_310.7s` - Block 0, n-back Level 0, Start bei 129.5s, 310.7s Dauer
- `block_01_nback_1_onset_440.2s_dur_218.2s` - Block 1, n-back Level 1, Start bei 440.2s, 218.2s Dauer

Dieses Format ermöglicht:
- **Präzise Zeitangaben** in Sekunden
- **Einfache Extraktion** von Timing-Informationen aus der Beschreibung
- **Kompatibilität** mit MNE-Annotations

## Verwendung

### Pipeline ausführen

```python
from pathlib import Path
from eeg_pipeline import EEGPipeline, create_default_config

# Konfiguration erstellen
config = create_default_config()
config.output_dir = Path("results/processed_with_annotations")

# Pipeline ausführen
pipeline = EEGPipeline(config)
sessions = pipeline.run(Path("data"))
```

### Annotierte Daten laden und analysieren

```python
import mne
from pathlib import Path

# FIF-Datei mit Annotationen laden
raw = mne.io.read_raw_fif("results/processed_with_annotations/Aliaa/indoor_processed_raw.fif")

# Alle Annotationen anzeigen
print(f"Anzahl Annotationen: {len(raw.annotations)}")
for i, desc in enumerate(raw.annotations.description):
    onset = raw.annotations.onset[i] 
    duration = raw.annotations.duration[i]
    print(f"{desc}")

# Beispiel Output:
# baseline_1_onset_0.0s_0smp_dur_121.0s_30243smp
# block_00_nback_0_onset_129.5s_32363smp_dur_310.7s_77682smp
# block_01_nback_1_onset_440.2s_110046smp_dur_218.2s_54551smp
```

### Spezifische Blöcke extrahieren

```python
def extract_blocks_by_nback(raw, n_back_level):
    """Extrahiert alle Blöcke eines bestimmten n-back Levels."""
    blocks = []
    
    for i, desc in enumerate(raw.annotations.description):
        if f"nback_{n_back_level}" in desc and "block_" in desc:
            onset = raw.annotations.onset[i]
            duration = raw.annotations.duration[i]
            
            # Block extrahieren
            block = raw.copy().crop(tmin=onset, tmax=onset + duration)
            blocks.append(block)
    
    return blocks

def extract_baselines(raw):
    """Extrahiert alle Baseline-Perioden."""
    baselines = []
    
    for i, desc in enumerate(raw.annotations.description):
        if "baseline" in desc:
            onset = raw.annotations.onset[i]
            duration = raw.annotations.duration[i]
            
            # Baseline extrahieren
            baseline = raw.copy().crop(tmin=onset, tmax=onset + duration)
            baselines.append(baseline)
    
    return baselines

# Alle n-back 2 Blöcke extrahieren
nback2_blocks = extract_blocks_by_nback(raw, 2)
print(f"Gefunden: {len(nback2_blocks)} n-back 2 Blöcke")

# Alle Baseline-Perioden extrahieren
baselines = extract_baselines(raw)
print(f"Gefunden: {len(baselines)} Baseline-Perioden")

# Ersten Block analysieren
if nback2_blocks:
    block = nback2_blocks[0]
    
    # Timing-Informationen aus Beschreibung parsen
    desc = block.info['description']
    # Beispiel: block_02_nback_2_onset_658.4s_164598smp_dur_213.9s_53470smp
    if "_onset_" in desc and "_dur_" in desc:
        parts = desc.split("_")
        onset_s = float([p for p in parts if p.endswith("s")][0][:-1])
        onset_smp = int([p for p in parts if p.endswith("smp")][0][:-3])
        duration_s = float([p for p in parts if p.endswith("s")][1][:-1])
        duration_smp = int([p for p in parts if p.endswith("smp")][1][:-3])
        
        print(f"Block Timing:")
        print(f"  Onset: {onset_s:.1f}s ({onset_smp} samples)")
        print(f"  Dauer: {duration_s:.1f}s ({duration_smp} samples)")
    
    psd, freqs = mne.time_frequency.psd_welch(block, fmin=1, fmax=40)
    # Weitere Analyse...
```

## Dateien und Struktur

### Neue Dateien
- `src/eeg_pipeline/marker_annotation.py` - Marker-Annotation Funktionalität
- `test_pipeline_annotations.py` - Test-Skript für neue Funktionalität  
- `analyze_annotated_blocks.py` - Beispiel-Analyse der annotierten Daten

### Modifizierte Dateien
- `src/eeg_pipeline/pipeline.py` - Erweitert um Annotation-Schritt
- `src/eeg_pipeline/data_loading.py` - Bugfix für Marker-Loading
- `src/eeg_pipeline/__init__.py` - Export der neuen Funktionen

## Output-Struktur

```
results/processed_with_annotations/
├── Aliaa/
│   ├── indoor_processed_raw.fif    # Mit n-back Annotationen
│   └── outdoor_processed_raw.fif   # Mit n-back Annotationen
├── Anita/
│   ├── indoor_processed_raw.fif
│   └── outdoor_processed_raw.fif
└── ...
```

## Technische Details

### N-back Erkennung

Die n-back Level werden durch Analyse der Sequenzen und Targets bestimmt:

- **n-back 0**: Baseline, erster Block
- **n-back 1**: Targets stimmen mit Position n-1 überein
- **n-back 2**: Targets stimmen mit Position n-2 überein  
- **n-back 3**: Targets stimmen mit Position n-3 überein

### Zeitkonversion

- **Marker-Timestamps**: Absolute Zeit in Sekunden
- **EEG-Daten**: Relative Zeit ab Aufnahme-Start
- **Annotationen**: Relative Zeit passend zu EEG-Daten

### Sampling Rate Konvention

Die Pipeline arbeitet direkt mit den tatsächlichen Sampling-Raten aus den XDF-Dateien und verwendet Zeitstempel in Sekunden für die Annotationen.

## Vorteile

1. **Automatisierung**: Keine manuelle Block-Identifikation nötig
2. **Präzision**: Exakte Zeit-Grenzen aus Marker-Stream
3. **Baseline-Integration**: Automatische Erkennung von Ruhephasen
4. **Einfaches Format**: Klare Zeitangaben in Sekunden
5. **Reproduzierbarkeit**: Identische Annotation bei jedem Pipeline-Lauf
6. **MNE-Kompatibilität**: Standard MNE Annotations für einfache Weiterverarbeitung
7. **Flexibilität**: Einfache Extraktion spezifischer Blöcke und Baselines für Analyse

## Beispiel-Workflow

```python
# 1. Pipeline ausführen
config = create_default_config()
config.output_dir = Path("results/annotated")
pipeline = EEGPipeline(config)
sessions = pipeline.run(Path("data"))

# 2. Daten laden
raw = mne.io.read_raw_fif("results/annotated/Participant/session.fif")

# 3. Baseline extrahieren
baselines = extract_baselines(raw)
baseline_power = []
for baseline in baselines:
    psd, freqs = mne.time_frequency.psd_welch(baseline)
    alpha_power = np.mean(psd[:, (freqs >= 8) & (freqs <= 12)])
    baseline_power.append(alpha_power)

# 4. Spezifische Analyse
high_load_blocks = extract_blocks_by_nback(raw, 3)  # n-back 3
low_load_blocks = extract_blocks_by_nback(raw, 0)   # n-back 0

# 5. Vergleichsanalyse
for block in high_load_blocks:
    # Timing-Informationen verfügbar
    desc = block.info['description']
    # Beispiel: block_04_nback_3_onset_1091.2s_dur_216.1s
    
    psd, freqs = mne.time_frequency.psd_welch(block)
    # Alpha/Beta Power, Connectivity, etc.
```

Diese Erweiterung macht die Pipeline deutlich leistungsfähiger für kognitive Workload-Studien und ermöglicht präzise, reproduzierbare Analysen sowohl der experimentellen Bedingungen als auch der Baseline-Perioden.
