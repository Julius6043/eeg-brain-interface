#!/usr/bin/env python3
"""
Generelles Debug-Script für XDF-Dateien
Überprüft alle XDF-Dateien im Projekt auf Datenqualität und Vollständigkeit
"""

from pathlib import Path
import sys
import traceback
from typing import Dict, List, Optional, Tuple
import pandas as pd

from eeg_pipeline.data_loading import DataLoadingConfig, load_xdf_safe, pick_streams, eeg_stream_to_raw


class XDFAnalyzer:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.config = DataLoadingConfig(max_channels=8, montage="standard_1020")
        self.results = []

    def analyze_xdf_file(self, xdf_path: Path) -> Dict:
        """Analysiert eine einzelne XDF-Datei"""
        result = {
            'file_path': str(xdf_path),
            'participant': self._extract_participant(xdf_path),
            'session': self._extract_session(xdf_path),
            'file_exists': xdf_path.exists(),
            'file_size_mb': 0,
            'streams_found': 0,
            'eeg_stream_found': False,
            'marker_stream_found': False,
            'eeg_channels': 0,
            'eeg_samples': 0,
            'eeg_duration_sec': 0,
            'eeg_sampling_rate': 0,
            'data_range_min': None,
            'data_range_max': None,
            'data_median': None,
            'has_nan_values': False,
            'has_inf_values': False,
            'load_successful': False,
            'raw_object_created': False,
            'error_message': None,
            'warnings': []
        }

        if not xdf_path.exists():
            result['error_message'] = "Datei existiert nicht"
            return result

        try:
            # Dateigröße
            result['file_size_mb'] = round(xdf_path.stat().st_size / (1024 * 1024), 2)

            # XDF laden
            streams, header = load_xdf_safe(xdf_path)
            if not streams:
                result['error_message'] = "XDF konnte nicht geladen werden"
                return result

            result['streams_found'] = len(streams)
            result['load_successful'] = True

            # Streams analysieren
            eeg_stream, marker_stream = pick_streams(streams)
            result['eeg_stream_found'] = eeg_stream is not None
            result['marker_stream_found'] = marker_stream is not None

            if not eeg_stream:
                result['warnings'].append("Kein EEG-Stream gefunden")
                return result

            # EEG-Stream Details
            info = eeg_stream.get("info", {})
            time_series = eeg_stream.get("time_series", [])

            if len(time_series) == 0:
                result['warnings'].append("EEG time_series ist leer")
                return result

            import numpy as np
            data = np.array(time_series, dtype=float)

            result['eeg_channels'] = data.shape[1] if len(data.shape) > 1 else 0
            result['eeg_samples'] = data.shape[0] if len(data.shape) > 0 else 0

            # Sampling Rate
            nominal_srate = info.get("nominal_srate", [0])
            if isinstance(nominal_srate, list) and len(nominal_srate) > 0:
                result['eeg_sampling_rate'] = float(nominal_srate[0])
            else:
                result['eeg_sampling_rate'] = float(nominal_srate)

            if result['eeg_sampling_rate'] > 0:
                result['eeg_duration_sec'] = round(result['eeg_samples'] / result['eeg_sampling_rate'], 2)

            # Datenqualität prüfen
            if data.size > 0:
                # Transponieren für MNE-Format (channels x time)
                data_t = data.T
                result['data_range_min'] = float(np.nanmin(data_t))
                result['data_range_max'] = float(np.nanmax(data_t))
                result['data_median'] = float(np.nanmedian(data_t))
                result['has_nan_values'] = bool(np.isnan(data_t).any())
                result['has_inf_values'] = bool(np.isinf(data_t).any())

                # Versuche Raw-Objekt zu erstellen
                try:
                    raw = eeg_stream_to_raw(eeg_stream, self.config)
                    result['raw_object_created'] = True

                    if raw.n_times == 0:
                        result['warnings'].append("Raw-Objekt hat 0 Zeitpunkte")

                except Exception as e:
                    result['warnings'].append(f"Raw-Objekt Erstellung fehlgeschlagen: {str(e)}")
            else:
                result['warnings'].append("Keine Daten im EEG-Stream")

        except Exception as e:
            result['error_message'] = str(e)
            result['warnings'].append(f"Unerwarteter Fehler: {str(e)}")

        return result

    def _extract_participant(self, xdf_path: Path) -> str:
        """Extrahiert Teilnehmernamen aus dem Pfad"""
        parts = xdf_path.parts
        for part in parts:
            if part.startswith('sub-P'):
                return part.split('_')[-1] if '_' in part else part
        return "Unknown"

    def _extract_session(self, xdf_path: Path) -> str:
        """Extrahiert Session-Info aus dem Pfad"""
        parts = xdf_path.parts
        for part in parts:
            if part.startswith('ses-S'):
                return part
        return "Unknown"

    def analyze_all_files(self) -> List[Dict]:
        """Analysiert alle XDF-Dateien im Datenverzeichnis"""
        print(f"🔍 Durchsuche XDF-Dateien in: {self.data_dir}")
        print("=" * 80)

        xdf_files = list(self.data_dir.rglob("*.xdf"))
        print(f"📁 Gefundene XDF-Dateien: {len(xdf_files)}")
        print()

        for i, xdf_path in enumerate(xdf_files, 1):
            print(f"[{i}/{len(xdf_files)}] Analysiere: {xdf_path.name}")
            result = self.analyze_xdf_file(xdf_path)
            self.results.append(result)

            # Kurze Zusammenfassung pro Datei
            status = "✅" if result['load_successful'] and result['raw_object_created'] else "❌"
            print(f"  {status} {result['participant']} | {result['session']} | "
                  f"{result['eeg_samples']} samples | {result['eeg_duration_sec']}s")

            if result['error_message']:
                print(f"    ❌ Fehler: {result['error_message']}")

            if result['warnings']:
                for warning in result['warnings']:
                    print(f"    ⚠️  {warning}")
            print()

        return self.results

    def generate_summary_report(self) -> str:
        """Erstellt einen Zusammenfassungsbericht"""
        if not self.results:
            return "Keine Ergebnisse verfügbar"

        total_files = len(self.results)
        successful_loads = sum(1 for r in self.results if r['load_successful'])
        successful_raw = sum(1 for r in self.results if r['raw_object_created'])
        files_with_errors = sum(1 for r in self.results if r['error_message'])
        files_with_warnings = sum(1 for r in self.results if r['warnings'])

        # Teilnehmer-Session Matrix
        participants = {}
        for result in self.results:
            participant = result['participant']
            session = result['session']
            success = result['raw_object_created']

            if participant not in participants:
                participants[participant] = {}
            participants[participant][session] = success

        report = f"""
📊 XDF-DATEIEN ANALYSE BERICHT
{'=' * 50}

📈 ÜBERSICHT:
  • Gesamte XDF-Dateien: {total_files}
  • Erfolgreich geladen: {successful_loads} ({successful_loads / total_files * 100:.1f}%)
  • Raw-Objekte erstellt: {successful_raw} ({successful_raw / total_files * 100:.1f}%)
  • Dateien mit Fehlern: {files_with_errors}
  • Dateien mit Warnungen: {files_with_warnings}

👥 TEILNEHMER-SESSION MATRIX:
"""

        for participant, sessions in sorted(participants.items()):
            indoor = sessions.get('ses-S001', None)
            outdoor = sessions.get('ses-S002', None)

            indoor_status = "✅" if indoor is True else "❌" if indoor is False else "➖"
            outdoor_status = "✅" if outdoor is True else "❌" if outdoor is False else "➖"

            report += f"  {participant:10} | Indoor: {indoor_status} | Outdoor: {outdoor_status}\n"

        # Problematische Dateien
        problematic = [r for r in self.results if r['error_message'] or r['warnings'] or not r['raw_object_created']]
        if problematic:
            report += f"\n🚨 PROBLEMATISCHE DATEIEN ({len(problematic)}):\n"
            for result in problematic:
                report += f"  • {result['participant']} ({result['session']}):\n"
                if result['error_message']:
                    report += f"    ❌ {result['error_message']}\n"
                for warning in result['warnings']:
                    report += f"    ⚠️  {warning}\n"

        # Datenqualitäts-Statistiken
        valid_results = [r for r in self.results if r['eeg_samples'] > 0]
        if valid_results:
            avg_duration = sum(r['eeg_duration_sec'] for r in valid_results) / len(valid_results)
            total_duration = sum(r['eeg_duration_sec'] for r in valid_results)

            report += f"\n📊 DATENQUALITÄT:\n"
            report += f"  • Durchschnittliche Session-Dauer: {avg_duration:.1f} Sekunden\n"
            report += f"  • Gesamte Aufnahmezeit: {total_duration:.1f} Sekunden ({total_duration / 60:.1f} Minuten)\n"

        return report

    def save_detailed_csv(self, output_path: Path):
        """Speichert detaillierte Ergebnisse als CSV"""
        if not self.results:
            print("Keine Ergebnisse zum Speichern verfügbar")
            return

        df = pd.DataFrame(self.results)

        # Listen zu Strings konvertieren für CSV
        df['warnings'] = df['warnings'].apply(lambda x: '; '.join(x) if x else '')

        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"📄 Detaillierte Ergebnisse gespeichert: {output_path}")


def main():
    """Hauptfunktion für das Kommandozeilen-Tool"""
    print("🧠 EEG XDF-Dateien Debug-Analyzer")
    print("=" * 50)

    # Projekt-Root finden - suche nach pyproject.toml oder README.md
    current_dir = Path.cwd()
    project_root = None

    # Versuche das Projekt-Root zu finden, indem wir nach oben traversieren
    search_dir = current_dir
    max_levels = 10  # Verhindere Endlosschleife

    for _ in range(max_levels):
        # Prüfe auf typische Projekt-Root Indikatoren
        if (search_dir / "pyproject.toml").exists() or \
           (search_dir / "README.md").exists() and (search_dir / "data").exists():
            project_root = search_dir
            break

        # Wenn wir bereits im Root-Verzeichnis sind, stoppe
        if search_dir.parent == search_dir:
            break
            
        search_dir = search_dir.parent

    # Fallback: Verwende aktuelles Verzeichnis wenn kein Projekt-Root gefunden
    if project_root is None:
        print(f"⚠️  Projekt-Root nicht automatisch gefunden, verwende: {current_dir}")
        project_root = current_dir
    else:
        print(f"✅ Projekt-Root gefunden: {project_root}")

    data_dir = project_root / "data"
    results_dir = project_root / "results"

    results_dir.mkdir(exist_ok=True)

    if not data_dir.exists():
        print(f"❌ Datenverzeichnis nicht gefunden: {data_dir}")
        print(f"💡 Aktuelles Verzeichnis: {current_dir}")
        print(f"💡 Projekt-Root: {project_root}")
        print(f"💡 Verzeichnisstruktur prüfen:")
        print(f"    - {data_dir} (erwartet)")
        if project_root.exists():
            subdirs = [d.name for d in project_root.iterdir() if d.is_dir()]
            print(f"    - Verfügbare Verzeichnisse: {subdirs}")
        return 1

    # Analyzer starten
    analyzer = XDFAnalyzer(data_dir)

    try:
        # Alle Dateien analysieren
        results = analyzer.analyze_all_files()

        # Bericht erstellen und anzeigen
        report = analyzer.generate_summary_report()
        print(report)

        # Ergebnisse speichern
        csv_path = results_dir / "xdf_analysis_report.csv"
        analyzer.save_detailed_csv(csv_path)

        # Textbericht speichern
        report_path = results_dir / "xdf_analysis_summary.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📄 Zusammenfassungsbericht gespeichert: {report_path}")

        return 0

    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
