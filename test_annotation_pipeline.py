#!/usr/bin/env python3
"""Test script für die erweiterte EEG-Pipeline mit Block-Annotation.

Dieses Script testet die neue Annotation-Funktionalität der Pipeline
mit einem kleinen Subset der verfügbaren Daten.
"""

import sys
from pathlib import Path

# Zur Verfügung stehenden Pfad hinzufügen
sys.path.insert(0, str(Path(__file__).parent / "src"))

from eeg_pipeline.pipeline import EEGPipeline, create_default_config


def test_annotation_pipeline():
    """Teste die erweiterte Pipeline mit einem Teilnehmer."""

    # Konfiguration erstellen
    config = create_default_config()
    config.output_dir = Path("results/test_annotated")

    print("=== Test der erweiterten EEG-Pipeline ===")
    print(f"✓ Konfiguration erstellt")
    print(f"  - Data Loading: {config.data_loading.max_channels} Kanäle")
    print(
        f"  - Preprocessing: {config.preprocessing.l_freq}-{config.preprocessing.h_freq} Hz"
    )
    print(f"  - Annotation: {config.annotation.annotation_prefix} prefix")
    print(f"  - Difficulty Extractor: {config.annotation.difficulty_extractor_path}")
    print(f"  - Output: {config.output_dir}")

    # Pipeline erstellen und testen
    pipeline = EEGPipeline(config)

    # Test nur mit einem kleinen Subset (erste 2 Teilnehmer)
    data_dir = Path("data")
    if not data_dir.exists():
        print(f"❌ Datenverzeichnis nicht gefunden: {data_dir}")
        return False

    print(f"\n=== Pipeline-Test starten ===")
    try:
        sessions = pipeline.run(data_dir)
        print(f"\n✅ Pipeline erfolgreich abgeschlossen!")
        print(f"   Verarbeitete Sessions: {len(sessions)}")

        # Überprüfe, ob Annotationen hinzugefügt wurden
        for session in sessions[:2]:  # Nur erste 2 Sessions prüfen
            print(f"\n--- Session: {session.participant_name} ---")

            if session.indoor_session and session.indoor_session.annotations:
                n_annotations = len(session.indoor_session.annotations)
                nback_annotations = [
                    ann
                    for ann in session.indoor_session.annotations.description
                    if "nback" in ann
                ]
                print(
                    f"  Indoor: {n_annotations} Annotationen gesamt, {len(nback_annotations)} N-Back Blöcke"
                )
                if nback_annotations:
                    print(f"  Beispiel: {nback_annotations[0]}")

            if session.outdoor_session and session.outdoor_session.annotations:
                n_annotations = len(session.outdoor_session.annotations)
                nback_annotations = [
                    ann
                    for ann in session.outdoor_session.annotations.description
                    if "nback" in ann
                ]
                print(
                    f"  Outdoor: {n_annotations} Annotationen gesamt, {len(nback_annotations)} N-Back Blöcke"
                )
                if nback_annotations:
                    print(f"  Beispiel: {nback_annotations[0]}")

        return True

    except Exception as e:
        print(f"❌ Pipeline-Fehler: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_annotation_pipeline()
    if success:
        print("\n🎉 Test erfolgreich abgeschlossen!")
    else:
        print("\n💥 Test fehlgeschlagen!")
        sys.exit(1)
