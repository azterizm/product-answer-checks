"""Made-up products and their questions, for test 2's training data (docs/METHOD2.md §1 and §2).

No Icecat sheet is read here: the Open Content License (1.4, §10) bars machine learning training
on its content. The categories are words of METHOD §1's in-scope list; every feature table (group,
name, unit, values, in English and French) is written by hand from general product knowledge.
Brands and model names are invented. A product is a sheet in the shape ``sheets.py`` reads, so
test 1's code writes its listing and its questions.
"""

from __future__ import annotations

import functools
import itertools
import random
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from product_answer_checks import battery as bt
from product_answer_checks import sheets as sh
from product_answer_checks.selection import Candidate, b_feature

SEED = 20261104
PER_CATEGORY = 10
VALIDATION = 12
TYPES = "ABCDE"
BRANDS = (
    "Norvik",
    "Altaro",
    "Brevin",
    "Calvex",
    "Dorana",
    "Elvar",
    "Gravon",
    "Istra",
    "Jorvel",
    "Kalmo",
    "Lunex",
    "Mavira",
)

Values = tuple[tuple[str, str], ...]
Row = tuple[str, str, Values]  # group key, "English name|French name", values


def n(values: str, unit: str = "", fr_unit: str | None = None) -> Values:
    """Numbers with a unit, the French with a decimal comma: ``n("1.5 2", "kg")``."""
    fr = unit if fr_unit is None else fr_unit
    return tuple(
        (f"{v} {unit}".rstrip(), f"{v.replace('.', ',')} {fr}".rstrip()) for v in values.split()
    )


def span(lo: str, hi: str, step: str, unit: str = "", fr_unit: str | None = None) -> Values:
    """Every value from ``lo`` to ``hi`` by ``step``, as ``n`` writes them."""
    out, x = [], Decimal(lo)
    while x <= Decimal(hi):
        out.append(str(x))
        x += Decimal(step)
    return n(" ".join(out), unit, fr_unit)


def t(*pairs: str) -> Values:
    """Text values, ``"English|French"``, or one form for both."""
    return tuple((p.split("|")[0], p.split("|")[-1]) for p in pairs)


YN = t("Yes|Oui", "No|Non")
GROUPS = {
    "display": ("Display", "Écran"),
    "cpu": ("Processor", "Processeur"),
    "memory": ("Memory", "Mémoire"),
    "storage": ("Storage", "Stockage"),
    "graphics": ("Graphics", "Graphiques"),
    "battery": ("Battery", "Batterie"),
    "ports": ("Ports & interfaces", "Ports et interfaces"),
    "network": ("Networking", "Réseau"),
    "camera": ("Camera", "Caméra"),
    "audio": ("Audio", "Audio"),
    "software": ("Software", "Logiciel"),
    "keyboard": ("Keyboard", "Clavier"),
    "features": ("Features", "Caractéristiques"),
    "performance": ("Performance", "Performance"),
    "ergonomics": ("Ergonomics", "Ergonomie"),
    "power": ("Power", "Alimentation"),
    "energy": ("Energy management", "Gestion de l'énergie"),
    "design": ("Design", "Design"),
    "size": ("Weight & dimensions", "Poids et dimensions"),
}
COLOURS = t("Black|Noir", "Silver|Argent", "Grey|Gris", "White|Blanc", "Blue|Bleu")
SDR_1000H = "Energy consumption (SDR) per 1000 hours|Consommation d'énergie (SDR) par 1000 heures"
HDR_1000H = "Energy consumption (HDR) per 1000 hours|Consommation d'énergie (HDR) par 1000 heures"
WINDOWS = t("Windows 11 Home|Windows 11 Famille", "Windows 11 Pro|Windows 11 Professionnel")


@dataclass(frozen=True)
class Category:
    name: tuple[str, str]
    noun: tuple[str, str]
    series: tuple[str, ...]
    rows: tuple[Row, ...]

    @property
    def energy(self) -> bool:
        return any(name.startswith("Energy efficiency class") for _, name, _ in self.rows)


def _inches(*pairs: str) -> Values:
    """Screen sizes ``"35.6 14"``: centimetres, then inches in brackets."""
    out = []
    for pair in pairs:
        cm, inch = pair.split()
        out.append((f'{cm} cm ({inch}")', f'{cm.replace(".", ",")} cm ({inch.replace(".", ",")}")'))
    return tuple(out)


CATEGORIES = (
    Category(
        ("Laptops", "Ordinateurs portables"),
        ("laptop", "ordinateur portable"),
        ("Aero", "Vela", "Strato", "Pivot"),
        (
            (
                "display",
                "Display diagonal|Diagonale de l'écran",
                _inches("33.8 13.3", "35.6 14", "39.6 15.6", "40.6 16", "43.9 17.3"),
            ),
            (
                "display",
                "Display resolution|Résolution de l'écran",
                t("1920 x 1080 pixels", "1920 x 1200 pixels", "2560 x 1600 pixels"),
            ),
            ("display", "Refresh rate|Taux de rafraîchissement", n("60 90 120 144 165", "Hz")),
            ("display", "Display brightness|Luminosité de l'écran", n("250 300 400 500", "cd/m²")),
            (
                "cpu",
                "Processor base frequency|Fréquence de base du processeur",
                n("1.8 2.0 2.2 2.4 2.6 3.0 3.2", "GHz"),
            ),
            ("cpu", "Processor cores|Nombre de cœurs de processeur", n("4 6 8 10 12 14 16")),
            ("memory", "Internal memory|Mémoire interne", n("8 16 24 32 64", "GB", "Go")),
            (
                "memory",
                "Memory clock speed|Vitesse d'horloge de la mémoire",
                n("4800 5600 6400 7500", "MHz"),
            ),
            (
                "storage",
                "Total storage capacity|Capacité totale de stockage",
                n("256 512 1024 2048", "GB", "Go"),
            ),
            ("battery", "Battery capacity|Capacité de la batterie", n("45 54 60 70 80 99", "Wh")),
            ("battery", "Battery life|Autonomie de la batterie", span("8", "20", "1", "h")),
            ("ports", "USB Type-C ports|Nombre de ports USB Type-C", n("1 2 3")),
            ("ports", "HDMI ports|Nombre de ports HDMI", n("0 1")),
            ("network", "Wi-Fi standard|Norme Wi-Fi", t("Wi-Fi 6", "Wi-Fi 6E", "Wi-Fi 7")),
            ("camera", "Front camera resolution|Résolution de la caméra avant", n("1 2 5", "MP")),
            ("software", "Operating system installed|Système d'exploitation installé", WINDOWS),
            ("keyboard", "Numeric keypad|Pavé numérique", YN),
            ("keyboard", "Backlit keyboard|Clavier rétroéclairé", YN),
            ("design", "Colour|Couleur", COLOURS),
            ("size", "Width|Largeur", span("300", "385", "5", "mm")),
            ("size", "Depth|Profondeur", span("200", "260", "4", "mm")),
            ("size", "Height|Hauteur", span("14.5", "22.5", "0.5", "mm")),
            ("size", "Weight|Poids", n("1.1 1.25 1.4 1.55 1.7 1.9 2.2 2.5", "kg")),
        ),
    ),
    Category(
        ("Desktop PCs", "PC de bureau"),
        ("desktop PC", "PC de bureau"),
        ("Forge", "Tessera", "Atlas"),
        (
            (
                "cpu",
                "Processor base frequency|Fréquence de base du processeur",
                span("2.1", "3.8", "0.1", "GHz"),
            ),
            (
                "cpu",
                "Processor boost frequency|Fréquence turbo du processeur",
                span("4.2", "5.8", "0.1", "GHz"),
            ),
            ("cpu", "Processor cores|Nombre de cœurs de processeur", n("6 8 10 12 14 16 20 24")),
            ("memory", "Internal memory|Mémoire interne", n("8 16 32 64", "GB", "Go")),
            (
                "memory",
                "Maximum internal memory|Mémoire interne maximale",
                n("64 96 128 192", "GB", "Go"),
            ),
            ("memory", "Memory slots|Nombre d'emplacements mémoire", n("2 4")),
            (
                "storage",
                "Total storage capacity|Capacité totale de stockage",
                n("512 1024 2048 4096", "GB", "Go"),
            ),
            (
                "graphics",
                "Discrete graphics memory|Mémoire graphique dédiée",
                n("4 6 8 12 16", "GB", "Go"),
            ),
            ("ports", "USB Type-A ports|Nombre de ports USB Type-A", n("2 4 6 8")),
            ("ports", "USB Type-C ports|Nombre de ports USB Type-C", n("1 2")),
            ("ports", "HDMI ports|Nombre de ports HDMI", n("1 2")),
            ("ports", "DisplayPort outputs|Nombre de sorties DisplayPort", n("0 1 2 3")),
            ("network", "Ethernet data rate|Débit Ethernet", n("1000 2500", "Mbit/s")),
            ("network", "Wi-Fi standard|Norme Wi-Fi", t("Wi-Fi 6", "Wi-Fi 6E", "Wi-Fi 7")),
            ("power", "Power supply|Alimentation électrique", n("180 260 300 500 750", "W")),
            ("software", "Operating system installed|Système d'exploitation installé", WINDOWS),
            (
                "design",
                "Chassis type|Type de châssis",
                t("Tower|Tour", "Mini tower|Mini tour", "Small form factor|Petit format"),
            ),
            ("design", "Colour|Couleur", t("Black|Noir", "Grey|Gris", "White|Blanc")),
            ("size", "Width|Largeur", span("90", "200", "5", "mm")),
            ("size", "Depth|Profondeur", span("290", "420", "10", "mm")),
            ("size", "Height|Hauteur", span("290", "450", "10", "mm")),
            ("size", "Weight|Poids", span("4.5", "12", "0.5", "kg")),
        ),
    ),
    Category(
        ("Monitors", "Moniteurs"),
        ("monitor", "moniteur"),
        ("Clarion", "Prism", "Vista"),
        (
            (
                "display",
                "Display diagonal|Diagonale de l'écran",
                _inches("60.5 23.8", "68.6 27", "80 31.5", "86.4 34"),
            ),
            (
                "display",
                "Display resolution|Résolution de l'écran",
                t(
                    "1920 x 1080 pixels",
                    "2560 x 1440 pixels",
                    "3440 x 1440 pixels",
                    "3840 x 2160 pixels",
                ),
            ),
            ("display", "Refresh rate|Taux de rafraîchissement", n("60 75 100 144 165 240", "Hz")),
            ("display", "Response time|Temps de réponse", n("1 2 4 5", "ms")),
            ("display", "Display brightness|Luminosité de l'écran", n("250 300 350 400", "cd/m²")),
            (
                "display",
                "Contrast ratio (typical)|Rapport de contraste (typique)",
                t("1000:1", "3000:1"),
            ),
            ("display", "Panel type|Type de panneau", t("IPS", "VA", "OLED")),
            ("ports", "HDMI ports|Nombre de ports HDMI", n("1 2")),
            ("ports", "DisplayPort inputs|Nombre d'entrées DisplayPort", n("0 1 2")),
            ("ports", "USB-C power delivery|Puissance USB-C", n("15 65 90", "W")),
            ("ergonomics", "Height adjustment|Réglage de la hauteur", n("0 100 130 150", "mm")),
            ("ergonomics", "Tilt adjustment|Inclinaison réglable", YN),
            ("audio", "Built-in speakers|Haut-parleurs intégrés", YN),
            (
                "energy",
                "Energy efficiency class|Classe d'efficacité énergétique",
                t("D", "E", "F", "G"),
            ),
            (
                "energy",
                SDR_1000H,
                span("12", "35", "1", "kWh"),
            ),
            (
                "energy",
                "Power consumption (typical)|Consommation électrique (typique)",
                span("18", "45", "1", "W"),
            ),
            ("design", "Colour|Couleur", t("Black|Noir", "White|Blanc", "Silver|Argent")),
            ("size", "Width (with stand)|Largeur (avec pied)", span("540", "810", "10", "mm")),
            ("size", "Depth (with stand)|Profondeur (avec pied)", span("180", "250", "5", "mm")),
            ("size", "Height (with stand)|Hauteur (avec pied)", span("380", "580", "10", "mm")),
            ("size", "Weight (with stand)|Poids (avec pied)", span("3.5", "9.8", "0.1", "kg")),
        ),
    ),
    Category(
        ("TVs", "Téléviseurs"),
        ("TV", "téléviseur"),
        ("Lumen", "Horizon", "Aurea"),
        (
            (
                "display",
                "Display diagonal|Diagonale de l'écran",
                _inches("109.2 43", "127 50", "139.7 55", "165.1 65", "190.5 75"),
            ),
            (
                "display",
                "Display resolution|Résolution de l'écran",
                t("1920 x 1080 pixels", "3840 x 2160 pixels"),
            ),
            ("display", "Display technology|Technologie d'affichage", t("LED", "QLED", "OLED")),
            (
                "display",
                "Native refresh rate|Taux de rafraîchissement natif",
                n("50 60 100 120", "Hz"),
            ),
            ("display", "HDR formats|Formats HDR", t("HDR10, HLG", "HDR10, HDR10+, HLG")),
            ("features", "Smart TV|Smart TV", YN),
            ("audio", "RMS rated power|Puissance nominale RMS", n("20 30 40 60", "W")),
            ("audio", "Number of speakers|Nombre de haut-parleurs", n("2 4")),
            ("ports", "HDMI ports|Nombre de ports HDMI", n("2 3 4")),
            ("ports", "USB ports|Nombre de ports USB", n("1 2")),
            ("network", "Wi-Fi|Wi-Fi", YN),
            (
                "energy",
                "Energy efficiency class (SDR)|Classe d'efficacité énergétique (SDR)",
                t("E", "F", "G"),
            ),
            (
                "energy",
                SDR_1000H,
                span("60", "160", "1", "kWh"),
            ),
            (
                "energy",
                HDR_1000H,
                span("90", "250", "1", "kWh"),
            ),
            (
                "energy",
                "Power consumption (standby)|Consommation électrique (veille)",
                n("0.3 0.5", "W"),
            ),
            ("design", "Colour|Couleur", t("Black|Noir", "Silver|Argent")),
            ("size", "Width (without stand)|Largeur (sans pied)", span("960", "1680", "10", "mm")),
            ("size", "Height (without stand)|Hauteur (sans pied)", span("560", "960", "10", "mm")),
            ("size", "Depth (without stand)|Profondeur (sans pied)", span("25", "90", "1", "mm")),
            ("size", "Weight (without stand)|Poids (sans pied)", span("6.5", "32", "0.5", "kg")),
        ),
    ),
    Category(
        ("Smartphones", "Smartphones"),
        ("smartphone", "smartphone"),
        ("Nova", "Echo", "Pulse"),
        (
            (
                "display",
                "Display diagonal|Diagonale de l'écran",
                _inches("15.5 6.1", "16 6.3", "17 6.7", "17.3 6.8"),
            ),
            (
                "display",
                "Display resolution|Résolution de l'écran",
                t("1080 x 2400 pixels", "1170 x 2532 pixels", "1440 x 3120 pixels"),
            ),
            ("display", "Refresh rate|Taux de rafraîchissement", n("60 90 120", "Hz")),
            (
                "cpu",
                "Processor frequency|Fréquence du processeur",
                span("2.4", "3.4", "0.1", "GHz"),
            ),
            ("cpu", "Processor cores|Nombre de cœurs de processeur", n("6 8 10")),
            ("memory", "RAM capacity|Capacité de la RAM", n("4 6 8 12 16", "GB", "Go")),
            (
                "storage",
                "Internal storage capacity|Capacité de stockage interne",
                n("64 128 256 512", "GB", "Go"),
            ),
            (
                "camera",
                "Rear camera resolution|Résolution de la caméra arrière",
                n("12 48 50 64 108 200", "MP"),
            ),
            (
                "camera",
                "Front camera resolution|Résolution de la caméra avant",
                n("8 10 12 16 32", "MP"),
            ),
            (
                "battery",
                "Battery capacity|Capacité de la batterie",
                span("3500", "5500", "100", "mAh"),
            ),
            ("battery", "Fast charging power|Puissance de charge rapide", n("18 25 33 45 65", "W")),
            ("battery", "Wireless charging|Charge sans fil", YN),
            ("network", "5G|5G", YN),
            ("network", "NFC|NFC", YN),
            ("features", "Dual SIM|Double SIM", YN),
            ("features", "IP code|Indice de protection", t("IP54", "IP67", "IP68")),
            ("design", "Colour|Couleur", COLOURS),
            ("size", "Width|Largeur", span("70", "78", "0.5", "mm")),
            ("size", "Height|Hauteur", span("146", "165", "0.5", "mm")),
            ("size", "Depth|Profondeur", span("7.4", "9.2", "0.1", "mm")),
            ("size", "Weight|Poids", span("160", "230", "1", "g")),
        ),
    ),
    Category(
        ("Tablets", "Tablettes"),
        ("tablet", "tablette"),
        ("Folio", "Slate", "Canvas"),
        (
            (
                "display",
                "Display diagonal|Diagonale de l'écran",
                _inches("25.6 10.1", "27.9 11", "31.5 12.4", "33 13"),
            ),
            (
                "display",
                "Display resolution|Résolution de l'écran",
                t(
                    "1920 x 1200 pixels",
                    "2000 x 1200 pixels",
                    "2560 x 1600 pixels",
                    "2800 x 1752 pixels",
                ),
            ),
            ("display", "Refresh rate|Taux de rafraîchissement", n("60 90 120 144", "Hz")),
            (
                "cpu",
                "Processor frequency|Fréquence du processeur",
                span("2.0", "3.2", "0.1", "GHz"),
            ),
            ("memory", "RAM capacity|Capacité de la RAM", n("4 6 8 12", "GB", "Go")),
            (
                "storage",
                "Internal storage capacity|Capacité de stockage interne",
                n("64 128 256 512", "GB", "Go"),
            ),
            (
                "storage",
                "Maximum memory card size|Taille maximale de la carte mémoire",
                n("512 1024", "GB", "Go"),
            ),
            (
                "camera",
                "Rear camera resolution|Résolution de la caméra arrière",
                n("8 13 50", "MP"),
            ),
            ("camera", "Front camera resolution|Résolution de la caméra avant", n("5 8 12", "MP")),
            (
                "battery",
                "Battery capacity|Capacité de la batterie",
                span("6000", "11200", "200", "mAh"),
            ),
            ("battery", "Battery life|Autonomie de la batterie", n("10 12 13 15", "h")),
            ("battery", "Charging power|Puissance de charge", n("15 18 25 45", "W")),
            ("features", "Stylus included|Stylet inclus", YN),
            ("network", "4G|4G", YN),
            ("design", "Colour|Couleur", t("Grey|Gris", "Silver|Argent", "Blue|Bleu")),
            ("size", "Width|Largeur", span("245", "290", "1", "mm")),
            ("size", "Height|Hauteur", span("155", "190", "1", "mm")),
            ("size", "Depth|Profondeur", span("5.5", "8", "0.1", "mm")),
            ("size", "Weight|Poids", span("450", "650", "5", "g")),
        ),
    ),
    Category(
        ("Headphones", "Casques"),
        ("headphones", "casque"),
        ("Tempo", "Quiet", "Arc"),
        (
            (
                "audio",
                "Form factor|Facteur de forme",
                t("Over-ear|Circum-aural", "On-ear|Supra-aural", "In-ear|Intra-auriculaire"),
            ),
            ("audio", "Driver diameter|Diamètre du haut-parleur", n("6 10 11 30 40 50", "mm")),
            ("audio", "Frequency range|Plage de fréquences", t("20 - 20000 Hz", "10 - 40000 Hz")),
            ("audio", "Impedance|Impédance", t("16 Ω", "32 Ω", "38 Ω", "64 Ω")),
            ("audio", "Sensitivity|Sensibilité", span("95", "110", "1", "dB")),
            ("features", "Active noise cancelling|Réduction de bruit active", YN),
            ("features", "Microphone|Microphone", YN),
            ("features", "Foldable|Pliable", YN),
            ("network", "Bluetooth version|Version Bluetooth", t("5.0|5,0", "5.2|5,2", "5.3|5,3")),
            ("network", "Bluetooth range|Portée Bluetooth", n("10 15 20", "m")),
            ("battery", "Battery life|Autonomie de la batterie", span("20", "60", "2", "h")),
            ("battery", "Charging time|Temps de charge", n("1.5 2 2.5 3", "h")),
            (
                "battery",
                "Battery capacity|Capacité de la batterie",
                span("300", "900", "20", "mAh"),
            ),
            ("ports", "Charging port|Port de charge", t("USB Type-C")),
            ("design", "Colour|Couleur", t("Black|Noir", "White|Blanc", "Blue|Bleu")),
            ("size", "Weight|Poids", span("45", "320", "5", "g")),
        ),
    ),
    Category(
        ("Washing machines", "Lave-linge"),
        ("washing machine", "lave-linge"),
        ("Aqua", "Cascade", "Tide"),
        (
            (
                "features",
                "Loading type|Type de chargement",
                t("Front-load|Frontal", "Top-load|Par le dessus"),
            ),
            ("performance", "Washing capacity|Capacité de lavage", n("7 8 9 10 11", "kg")),
            (
                "performance",
                "Maximum spin speed|Vitesse d'essorage maximale",
                n("1200 1400 1600", "RPM", "tr/min"),
            ),
            (
                "performance",
                "Spin-drying efficiency class|Classe d'efficacité d'essorage",
                t("A", "B", "C"),
            ),
            (
                "performance",
                "Noise level (spinning)|Niveau sonore (essorage)",
                span("70", "78", "1", "dB"),
            ),
            (
                "features",
                "Washing programmes|Nombre de programmes de lavage",
                span("12", "20", "1"),
            ),
            (
                "performance",
                "Eco 40-60 programme duration|Durée du programme éco 40-60",
                span("190", "240", "5", "min"),
            ),
            (
                "performance",
                "Water consumption per cycle|Consommation d'eau par cycle",
                span("45", "60", "1", "L"),
            ),
            ("features", "Delay start timer|Départ différé", YN),
            ("features", "Steam function|Fonction vapeur", YN),
            ("features", "Child lock|Sécurité enfants", YN),
            (
                "energy",
                "Energy efficiency class|Classe d'efficacité énergétique",
                t("A", "B", "C", "D"),
            ),
            (
                "energy",
                "Energy consumption per 100 cycles|Consommation d'énergie pour 100 cycles",
                span("45", "90", "1", "kWh"),
            ),
            ("design", "Colour|Couleur", t("White|Blanc", "Black|Noir")),
            ("size", "Width|Largeur", n("595 597 598 600", "mm")),
            ("size", "Depth|Profondeur", span("550", "650", "5", "mm")),
            ("size", "Height|Hauteur", n("845 847 850", "mm")),
            ("size", "Weight|Poids", span("60", "80", "1", "kg")),
        ),
    ),
    Category(
        ("Fridge-freezers", "Réfrigérateurs-congélateurs"),
        ("fridge-freezer", "réfrigérateur-congélateur"),
        ("Frost", "Polar", "Glacier"),
        (
            (
                "performance",
                "Total net capacity|Capacité nette totale",
                span("250", "420", "5", "L"),
            ),
            (
                "performance",
                "Fridge net capacity|Capacité nette du réfrigérateur",
                span("180", "300", "5", "L"),
            ),
            (
                "performance",
                "Freezer net capacity|Capacité nette du congélateur",
                span("70", "130", "2", "L"),
            ),
            (
                "features",
                "Freezer position|Position du congélateur",
                t("Bottom-placed|En bas", "Top-placed|En haut"),
            ),
            ("features", "No Frost|No Frost", YN),
            (
                "performance",
                "Freezing capacity|Pouvoir de congélation",
                t("4 kg/24h", "6 kg/24h", "8 kg/24h", "10 kg/24h"),
            ),
            ("performance", "Noise level|Niveau sonore", span("34", "42", "1", "dB")),
            ("performance", "Noise emission class|Classe d'émission sonore", t("A", "B", "C", "D")),
            ("features", "Climate class|Classe climatique", t("SN-T", "N-ST")),
            ("features", "Fridge shelves|Nombre d'étagères du réfrigérateur", n("3 4 5")),
            ("features", "Reversible door|Porte réversible", YN),
            (
                "energy",
                "Energy efficiency class|Classe d'efficacité énergétique",
                t("A", "B", "C", "D", "E"),
            ),
            (
                "energy",
                "Energy consumption per year|Consommation d'énergie par an",
                span("150", "330", "1", "kWh"),
            ),
            ("design", "Colour|Couleur", t("White|Blanc", "Silver|Argent", "Black|Noir")),
            ("size", "Width|Largeur", span("595", "700", "5", "mm")),
            ("size", "Depth|Profondeur", span("650", "710", "5", "mm")),
            ("size", "Height|Hauteur", span("1760", "2030", "10", "mm")),
            ("size", "Weight|Poids", span("60", "90", "1", "kg")),
        ),
    ),
    Category(
        ("Dishwashers", "Lave-vaisselle"),
        ("dishwasher", "lave-vaisselle"),
        ("Sparkle", "Rinse", "Clara"),
        (
            (
                "features",
                "Placement|Installation",
                t("Freestanding|Pose libre", "Fully built-in|Tout intégrable"),
            ),
            ("performance", "Place settings|Nombre de couverts", n("9 10 12 13 14 15")),
            ("features", "Washing programmes|Nombre de programmes de lavage", span("5", "9", "1")),
            ("features", "Temperature settings|Nombre de réglages de température", n("3 4 5")),
            (
                "performance",
                "Water consumption per cycle|Consommation d'eau par cycle",
                span("8.5", "12", "0.5", "L"),
            ),
            ("performance", "Noise level|Niveau sonore", span("39", "48", "1", "dB")),
            ("performance", "Noise emission class|Classe d'émission sonore", t("A", "B", "C")),
            (
                "performance",
                "Eco programme duration|Durée du programme éco",
                span("195", "240", "5", "min"),
            ),
            ("features", "Delay start timer|Départ différé", YN),
            ("features", "Half load|Demi-charge", YN),
            ("features", "Third rack|Troisième panier", YN),
            (
                "energy",
                "Energy efficiency class|Classe d'efficacité énergétique",
                t("A", "B", "C", "D", "E"),
            ),
            (
                "energy",
                "Energy consumption per 100 cycles|Consommation d'énergie pour 100 cycles",
                span("50", "95", "1", "kWh"),
            ),
            ("design", "Colour|Couleur", t("White|Blanc", "Silver|Argent", "Black|Noir")),
            ("size", "Width|Largeur", n("448 450 598 600", "mm")),
            ("size", "Depth|Profondeur", span("550", "600", "5", "mm")),
            ("size", "Height|Hauteur", span("815", "850", "5", "mm")),
            ("size", "Weight|Poids", span("32", "48", "1", "kg")),
        ),
    ),
    Category(
        ("Vacuum cleaners", "Aspirateurs"),
        ("vacuum cleaner", "aspirateur"),
        ("Swift", "Cyclo", "Breeze"),
        (
            (
                "features",
                "Product type|Type de produit",
                t("Cordless stick|Balai sans fil", "Canister|Traîneau", "Upright|Vertical"),
            ),
            ("performance", "Dust capacity|Capacité de poussière", span("0.4", "2.5", "0.1", "L")),
            ("performance", "Suction power|Puissance d'aspiration", span("100", "250", "10", "AW")),
            ("power", "Rated power|Puissance nominale", span("250", "900", "50", "W")),
            ("performance", "Noise level|Niveau sonore", span("72", "82", "1", "dB")),
            ("battery", "Battery voltage|Tension de la batterie", n("18 21.6 25.2", "V")),
            ("battery", "Run time|Autonomie", span("30", "70", "5", "min")),
            ("battery", "Charging time|Temps de charge", n("3.5 4 4.5 5", "h")),
            ("features", "Bagless|Sans sac", YN),
            ("features", "HEPA filtration|Filtration HEPA", YN),
            ("features", "Nozzles included|Nombre d'embouts inclus", n("2 3 4 5")),
            ("features", "Wall mounted charging|Charge murale", YN),
            ("design", "Colour|Couleur", t("Grey|Gris", "Blue|Bleu", "Red|Rouge")),
            ("size", "Width|Largeur", span("230", "300", "5", "mm")),
            ("size", "Depth|Profondeur", span("200", "320", "5", "mm")),
            ("size", "Height|Hauteur", span("1100", "1250", "10", "mm")),
            ("size", "Weight|Poids", span("1.5", "6.5", "0.1", "kg")),
        ),
    ),
    Category(
        ("Coffee machines", "Machines à café"),
        ("coffee machine", "machine à café"),
        ("Crema", "Barista", "Ristretto"),
        (
            (
                "features",
                "Product type|Type de produit",
                t(
                    "Espresso machine|Machine à expresso",
                    "Bean to cup|Machine à grains",
                    "Capsule machine|Machine à capsules",
                ),
            ),
            (
                "performance",
                "Water tank capacity|Capacité du réservoir d'eau",
                span("0.8", "2.0", "0.1", "L"),
            ),
            (
                "performance",
                "Bean container capacity|Capacité du réservoir à grains",
                span("200", "300", "10", "g"),
            ),
            ("performance", "Pump pressure|Pression de la pompe", n("15 19 20", "bar")),
            ("power", "Power|Puissance", span("1200", "1600", "50", "W")),
            ("performance", "Cups at once|Nombre de tasses simultanées", n("1 2")),
            ("performance", "Grind settings|Réglages de mouture", span("3", "15", "1")),
            ("performance", "Heat-up time|Temps de chauffe", span("25", "60", "5", "s")),
            ("features", "Built-in grinder|Moulin intégré", YN),
            ("features", "Milk frother|Mousseur à lait", YN),
            ("features", "Removable water tank|Réservoir d'eau amovible", YN),
            ("features", "Auto power off|Arrêt automatique", YN),
            ("features", "Programmable|Programmable", YN),
            ("design", "Colour|Couleur", t("Black|Noir", "Silver|Argent", "Red|Rouge")),
            ("size", "Width|Largeur", span("140", "280", "5", "mm")),
            ("size", "Depth|Profondeur", span("250", "440", "5", "mm")),
            ("size", "Height|Hauteur", span("280", "380", "5", "mm")),
            ("size", "Weight|Poids", span("2.5", "9.5", "0.1", "kg")),
        ),
    ),
)
IDS = tuple(range(1, len(CATEGORIES) * PER_CATEGORY + 1))


def category_of(pid: int) -> Category:
    return CATEGORIES[(pid - 1) // PER_CATEGORY]


def _feature_id(category: int, row: int, name: str) -> str:
    """Test 1's ids where its code looks for them (weight, energy class); else this table's."""
    if name.startswith("Weight"):
        return sh.WEIGHT
    if name.startswith("Energy efficiency class"):
        return sh.ENERGY_CLASS
    return f"x{category:02d}{row:02d}"


@functools.cache
def _draw(pid: int) -> tuple[str, str, tuple[int, ...]]:
    """A product's brand, model name and one value index per feature row."""
    category = category_of(pid)
    rng = random.Random(SEED + pid)
    brand = rng.choice(BRANDS)
    model = f"{rng.choice(category.series)} {rng.randint(10, 99) * 10}"
    picks = tuple(rng.randrange(len(values)) for _, _, values in category.rows)
    return brand, model, picks


def sheet(pid: int, lang: str) -> dict[str, Any]:
    """The product's sheet in one language, in the shape ``sheets.py`` reads."""
    k = 1 if lang == "fr" else 0
    c = (pid - 1) // PER_CATEGORY
    category = CATEGORIES[c]
    brand, model, picks = _draw(pid)
    groups: dict[str, list[dict[str, Any]]] = {}
    values = []
    for row, ((group, names, choices), pick) in enumerate(zip(category.rows, picks, strict=True)):
        name = names.split("|")[k]
        value = choices[pick][k]
        values.append(value)
        groups.setdefault(GROUPS[group][k], []).append(
            {
                "Feature": {"ID": _feature_id(c, row, names), "Name": {"Value": name}},
                "PresentationValue": value,
            }
        )
    title = f"{brand} {model} {category.noun[k]}"
    return {
        "msg": "OK",
        "data": {
            "GeneralInfo": {
                "Title": title,
                "Brand": brand,
                "Category": {"Name": {"Value": category.name[k]}},
                "TitleInfo": {"GeneratedLocalTitle": {"Value": title, "Language": lang.upper()}},
                "SummaryDescription": {
                    "ShortSummaryDescription": ", ".join([f"{brand} {model}", *values[:4]])
                },
            },
            "FeaturesGroups": [
                {"FeatureGroup": {"Name": {"Value": g}}, "Features": fs} for g, fs in groups.items()
            ],
        },
    }


def split() -> dict[int, str]:
    """Train or validation, by product (METHOD2 §1)."""
    ids = list(IDS)
    random.Random(SEED).shuffle(ids)
    held = set(ids[:VALIDATION])
    return {pid: "validation" if pid in held else "train" for pid in IDS}


def question_seed(pid: int, type_: str) -> int:
    return SEED + 1000 * pid + TYPES.index(type_)


def _candidate(pid: int) -> Candidate:
    en = sheet(pid, "en")
    return Candidate(
        pid,
        sh.category(en),
        sh.brand(en),
        energy=sh.energy_features(en) is not None,
        weight=sh.feature(en, sh.WEIGHT) is not None,
        numeric=sh.numeric_features(en),
    )


def triples() -> list[tuple[int, list[int], str]]:
    """Type B: in product order, each product with the first pair of later ones of its category and
    split that test 1's rule accepts; each product in at most one triple."""
    where = split()
    out: list[tuple[int, list[int], str]] = []
    used: set[int] = set()
    for pid in IDS:
        if pid in used:
            continue
        others = [
            o
            for o in IDS
            if o > pid
            and o not in used
            and category_of(o) is category_of(pid)
            and where[o] == where[pid]
        ]
        for pair in itertools.combinations(others, 2):
            three: Sequence[Candidate] = [_candidate(p) for p in (pid, *pair)]
            fid = b_feature(three, question_seed(pid, "B"))
            if fid is not None:
                out.append((pid, list(pair), fid))
                used.update((pid, *pair))
                break
    return out


def questions() -> list[bt.Question]:
    """Every training and validation question, both languages (METHOD2 §2)."""
    out: list[bt.Question] = []
    for pid in IDS:
        types = "ACDE" if category_of(pid).energy else "ACD"
        for type_ in types:
            seed = question_seed(pid, type_)
            out += bt.single_questions(f"T{pid:03d}{type_}", type_, pid, sheet, seed=seed)
    for pid, companions, fid in triples():
        seed = question_seed(pid, "B")
        out += bt.budget_questions(f"T{pid:03d}B", pid, companions, fid, sheet, seed=seed)
    return out
