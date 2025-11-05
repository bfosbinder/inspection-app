# Inspection App

A simple PyQt6 app to view a blueprint (PDF or image), jump to defined hotspots from a CSV, and pick new hotspots directly on the print.

## Requirements

- Python 3.10+ (tested with 3.14)
- Packages:
  - PyQt6
  - PyMuPDF (for PDF support)

You can install from `requirements.txt`.

## Setup (Linux, bash)

```bash
# from the project folder
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Run

```bash
# activate the venv if not already
source .venv/bin/activate
python main.py
```

- On first run, if `blueprint.pdf` (or a .png/.jpg) is not present, use "Open Blueprint" to choose one.
- If `hotspots.csv` is missing, the app will create a sample.

## Troubleshooting

- Module not found (PyQt6 or PyMuPDF): make sure your venv is active (`source .venv/bin/activate`) and reinstall: `pip install -r requirements.txt`.
- PDF opens error: install PyMuPDF: `pip install PyMuPDF`.
- Wayland windowing issues: if running on Wayland, try `QT_QPA_PLATFORM=xcb python main.py`.

## Notes

- Hotspots are stored in pixel coordinates of the rendered page. If you change render scale, existing hotspots may shift slightly.
- Picking a rectangle prompts for an ID and zoom; it appends to `hotspots.csv`.
