## Forschungskontext für die EEG‑Workload‑Dekodierung

### Relevante EEG‑Merkmale

**Spektrale Marker.** In der Literatur zur mentalen Belastung sind vor
allem Oszillationen im Theta‑ (4–7 Hz), Alpha‑ (8–12 Hz) und
Beta‑Band (13–30 Hz) relevant. Bei steigender kognitiver Belastung
nimmt die Theta‑Leistung insbesondere in frontalen Regionen zu,
während die Alpha‑Leistung in parietalen Arealen abnimmt【634168958309391†L324-L340】.  Diese
Bandänderungen wurden in zahlreichen n‑back‑Studien bestätigt und
erscheinen robust gegenüber verschiedenen Gerätearten und Versuchsumgebungen.

**Event‑Related Potentials (ERP).** Neben anhaltenden Oszillationen
reagiert das Gehirn auch auf einzelne Reize mit charakteristischen
Potentialen. Das **P300**‑Potential ist ein positiver Ausschlag
ungefähr 250–500 ms nach einem seltenen oder bedeutsamen Stimulus【990978214535399†L150-L155】.
Bei steigender Primäraufgabe bzw. höherer mentaler Belastung nimmt die
Amplitude des P300 ab【293848482666781†L390-L423】【34620982738676†L125-L126】.  Die Latenz des Peaks kann
ebenfalls verschoben sein, was Hinweise auf die Verarbeitungsdauer
gibt.  In n‑back‑Paradigmen korreliert eine geringere P300‑Amplitude
mit einer höheren n‑back‑Stufe (z. B. 2‑back vs. 0‑back).

### Preprocessing und Datenvorbereitung

Ein zentraler Schritt ist die Vorverarbeitung der Rohdaten. Wir
verwenden einen **50 Hz Notch‑Filter** zur Unterdrückung des
Netzbrummens und einen **Bandpassfilter von 1–40 Hz**, um sowohl
langsames Driften als auch hochfrequente Muskelartefakte zu
unterdrücken【102020619966441†L182-L209】.  Eine **Durchschnittsreferenz** ist bei
studienübergreifenden Vergleichen üblich und kann mit MNE über
`set_eeg_reference('average')` realisiert werden【296520584149078†L465-L477】.  Die
Vorverarbeitung entscheidet maßgeblich über die Datenqualität;
robustere Ansätze wie die PREP‑Pipeline betonen die Erkennung
rauschanfälliger Kanäle und die Verwendung einer robusten
Referenzberechnung【473351643201165†L170-L183】.

Die kontinuierlichen Daten werden in **2‑s‑Fenster mit 50 % Überlappung**
unterteilt. Kürzere Fenster erhöhen die Anzahl der Trainingsbeispiele,
ohne die Schätzgenauigkeit der spektralen Merkmale signifikant zu
beeinträchtigen.  Die Markerdatei liefert Zeitstempel der Stimuli
(`onset_s`), aus denen Epochen für die ERP‑Analyse (−0,2 bis +0,8 s)
extrahiert werden.

### Feature‑Extraktion

1. **Bandpower:** Für jedes Fenster wird das Leistungsdichtespektrum
   mittels der Welch‑Methode berechnet. Anschließend wird die Leistung
   im Theta‑, Alpha‑ und Beta‑Band gemittelt. Diese Features werden
   pro Kanal erstellt und können optional zu globalen Kennzahlen
   (z. B. Frontalth‑Theta/Parietal‑Alpha‑Ratio) aggregiert werden.
2. **P300‑Merkmale:** Aus den Epochen um jeden Stimulus wird das ERP
   über ausgewählte Kanäle gemittelt. Im Zeitfenster 0,25–0,45 s
   (typische Latenz des P300) werden die **Peakamplitude**, die
   **Peaktivität** und der **Mittelwert** berechnet. Diese Merkmale
   erfassen sowohl die Größe als auch die zeitliche Dynamik des
   P300【990978214535399†L150-L155】.
3. **Merkmalsfusion:** Die spektralen und ERP‑Merkmale werden
   concatenated. Eine Ausrichtung erfolgt, indem jeder Epoche der
   nächstgelegene Spektralfenster‑Zeitpunkt zugeordnet wird.

### Modellwahl und Training

**Lineare Modelle.** Logistic Regression mit Elastic‑Net‑Regularisierung
oder ein linearer Support Vector Machine (SVM) bilden die Standard‑
Baselines. In der Literatur zu Mental‑Workload‑Dekodierung werden
diese Modelle häufig eingesetzt und erreichen mit Bandpower‑Features
hohe Genauigkeiten【634168958309391†L160-L168】.  Lineare Modelle sind robust
gegenüber Überanpassung, interpretierbar und eignen sich gut für
8‑Kanal‑EEGs mit wenigen Trainingsdaten.

Wir verwenden eine **leave‑one‑subject‑out** Kreuzvalidierung, um
Generalisation auf unbekannte Versuchspersonen zu testen. Die
metriken sind **Balanced Accuracy**, **Macro‑F1** und **ROC‑AUC**.
Klassengewichte kompensieren ungleiche Klassenverteilungen.  SVMs
werden per **Platt‑Scaling** kalibriert, um Wahrscheinlichkeiten für
AUC‑Berechnungen zu erhalten.

**Tiefe Lernverfahren.** Als optionaler Stretch‑Goal implementieren wir
**EEGNet**, ein kompaktes CNN mit depthwise und separable
Convolutions. Diese Architektur ist für EEG‑BCI‑Tasks konzipiert und
lernt interpretable Filter mit wenigen Parametern【74903913843087†L230-L241】.  EEGNet
zeigt in Studien vergleichbare oder bessere Leistung als klassische
Methoden, verlangt jedoch eine sorgfältige Vorverarbeitung und Daten-
Augmentation. In unserem Setting dient EEGNet vor allem als
Referenzpunkt für tiefe Lernverfahren.

### Ergebnisse und Ausblick

Der beigefügte Trainingsskript demonstriert, wie mit den oben
beschriebenen Merkmalen und Modellen ein belastbares Baseline‑System
aufgebaut werden kann.  Die Kombination aus spektralen und P300‑
Merkmalen führt zu einer ausgewogenen Darstellung der anhaltenden und
transienten neuralen Dynamik und ermöglicht eine zuverlässige
Unterscheidung verschiedener Workload‑Stufen.  Erweiterungen wie die
Integration von Artefaktkorrektur nach dem PREP‑Prinzip, die Nutzung
größerer Datensätze und der Einsatz von Self‑Supervised‑Pretraining
könnten die Leistung weiter steigern.