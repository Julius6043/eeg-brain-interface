#!/usr/bin/env python3
"""Test Script für die erweiterte EEG-Pipeline mit Marker-Annotation.

Dieses Skript testet die neue Marker-Annotation Funktionalität.
"""

from pathlib import Path
from eeg_pipeline.pipeline import EEGPipeline, create_default_config


def test_pipeline_with_annotations():
    """Testet die Pipeline mit Marker-Annotationen."""

    # Konfiguration erstellen
    config = create_default_config()
    config.output_dir = Path("results/processed_with_annotations")

    # Teste nur mit einem kleinen Subset falls gewünscht
    # config.data_loading.max_channels = 4

    # Pipeline erstellen und ausführen
    pipeline = EEGPipeline(config)

    data_dir = Path("data")
    if not data_dir.exists():
        print(f"Datenverzeichnis {data_dir} nicht gefunden!")
        return

    print("Testing EEG Pipeline with Marker Annotations...")

    try:
        sessions = pipeline.run(data_dir)

        # Überprüfe Ergebnisse
        print(f"\n=== PIPELINE RESULTS ===")
        print(f"Sessions verarbeitet: {len(sessions)}")

        for session in sessions:
            print(f"\nTeilnehmer: {session.participant_name}")

            if session.indoor_session:
                annotations = session.indoor_session.annotations
                print(f"  Indoor Session: {len(annotations)} Annotationen")
                for i, desc in enumerate(annotations.description):
                    onset = annotations.onset[i]
                    duration = annotations.duration[i]
                    print(f"    {desc}: {onset:.1f}s - {onset + duration:.1f}s")
            else:
                print("  Indoor Session: Nicht verfügbar")

            if session.outdoor_session:
                annotations = session.outdoor_session.annotations
                print(f"  Outdoor Session: {len(annotations)} Annotationen")
                for i, desc in enumerate(annotations.description):
                    onset = annotations.onset[i]
                    duration = annotations.duration[i]
                    print(f"    {desc}: {onset:.1f}s - {onset + duration:.1f}s")
            else:
                print("  Outdoor Session: Nicht verfügbar")

        print(f"\n✓ Pipeline erfolgreich abgeschlossen!")
        print(f"Ergebnisse gespeichert in: {config.output_dir}")

    except Exception as e:
        print(f"✗ Pipeline Fehler: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_pipeline_with_annotations()
