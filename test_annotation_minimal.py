#!/usr/bin/env python3
"""Minimal-Test für die Annotation-Funktionalität ohne Pipeline-Ausführung."""

import sys
from pathlib import Path
import pandas as pd

# Zur Verfügung stehenden Pfad hinzufügen
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_annotation_logic():
    """Teste nur die Annotation-Logik ohne MNE-Abhängigkeiten."""

    print("=== Test der Annotation-Logik ===")

    try:
        from eeg_pipeline.annotation import (
            parse_blocks_from_marker_csv,
            load_difficulty_extractor,
            auto_find_difficulty_extractor,
        )

        print("✓ Annotation-Modul importiert")
    except ImportError as e:
        print(f"❌ Import-Fehler: {e}")
        return False

    # Test 1: Difficulty Extractor finden
    extractor_path = auto_find_difficulty_extractor()
    print(f"✓ Difficulty Extractor Pfad: {extractor_path}")

    # Test 2: Sample Marker-CSV laden
    sample_csv = Path("data/sub-P001_jannik/ses-S001/marker_log_p1s1_Jannik.csv")
    if sample_csv.exists():
        print(f"✓ Sample CSV gefunden: {sample_csv}")

        try:
            df_markers = pd.read_csv(sample_csv)
            print(f"✓ CSV geladen: {len(df_markers)} Zeilen")
            print(f"  Spalten: {list(df_markers.columns)}")

            # Test 3: Block-Parsing
            blocks_df = parse_blocks_from_marker_csv(df_markers)
            print(f"✓ Blocks geparst: {len(blocks_df)} Blöcke gefunden")
            print(f"  Block-Nummern: {blocks_df['block_num'].tolist()}")
            print(f"  Erste Onset-Zeiten: {blocks_df['onset_s'].head(3).tolist()}")

        except Exception as e:
            print(f"❌ CSV-Verarbeitung fehlgeschlagen: {e}")
            return False
    else:
        print(f"⚠ Sample CSV nicht gefunden: {sample_csv}")

    # Test 4: Difficulty Extractor laden (falls vorhanden)
    if extractor_path:
        try:
            calculate_nvals = load_difficulty_extractor(extractor_path)
            if calculate_nvals:
                print("✓ Difficulty Extractor erfolgreich geladen")

                # Test mit Sample-Daten
                if "df_markers" in locals():
                    df_for_calc = pd.DataFrame(
                        {"marker": df_markers["marker"].astype(str)}
                    )
                    nvals = calculate_nvals(df_for_calc)
                    print(f"✓ N-Werte berechnet: {nvals}")

            else:
                print("⚠ Difficulty Extractor konnte nicht geladen werden")
        except Exception as e:
            print(f"❌ Difficulty Extractor Fehler: {e}")
            return False

    print("\n✅ Annotation-Logik-Test erfolgreich!")
    return True


def test_imports():
    """Teste nur die Imports ohne Ausführung."""

    print("=== Import-Test ===")

    modules_to_test = [
        "eeg_pipeline.annotation",
        "eeg_pipeline.data_loading",
        "eeg_pipeline.preprocessing",
        "eeg_pipeline.pipeline",
    ]

    success = True
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
            success = False

    return success


if __name__ == "__main__":
    print("🧪 Starte erweiterte Pipeline-Tests...\n")

    # Test 1: Imports
    import_success = test_imports()

    if import_success:
        # Test 2: Annotation-Logik
        logic_success = test_annotation_logic()

        if logic_success:
            print("\n🎉 Alle Tests erfolgreich!")
        else:
            print("\n💥 Annotation-Tests fehlgeschlagen!")
            sys.exit(1)
    else:
        print("\n💥 Import-Tests fehlgeschlagen!")
        print("Hinweis: Führe 'pip install -r requirements.txt' aus")
        sys.exit(1)
