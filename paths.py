"""Chemins du dépôt, relatifs à ce fichier : les scripts tournent depuis n'importe où."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LATEX = ROOT / "latex"
for _d in (RESULTS, FIGURES, LATEX):
    _d.mkdir(exist_ok=True)
