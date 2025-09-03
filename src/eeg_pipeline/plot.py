"""Plot Configuration Placeholder.

Dieses Modul ist aktuell ein Platzhalter für zukünftige Visualisierungs-
Konfigurationen (z.B. PSD, Zeitbereich, Topographien, Event-Dichten).

Erweiterungsideen:
    * bool Flags (plot_psd, plot_raw, plot_ica_components)
    * Pfade für Ausgabe (separates Verzeichnis je Teilnehmer)
    * Matplotlib / MNE-Report Integration
"""

from dataclasses import dataclass


@dataclass
class PlotConfig:
    """Leere Konfiguration – dient als Erweiterungshaken."""

    # Beispiel zukünftiger Parameter (auskommentiert):
    # plot_psd: bool = True
    # plot_raw: bool = False
    # save_formats: tuple[str, ...] = ("png",)
    pass
