🧩 InspectionApp — Blueprint Inspection & Ballooning Tool

InspectionApp is a standalone Python / PyQt6 desktop application for creating and performing inspection plans directly on digital blueprints.
It allows engineers and inspectors to “pick-on-print” dimensions, manage inspection methods, record results, and even run OCR on dimensional callouts — all within one searchable, persistent workspace.

✨ Features
📄 Blueprint Viewing

<img width="1915" height="1059" alt="image" src="https://github.com/user-attachments/assets/ae21b2f7-83a7-4e94-b2dc-f1efc6e39f94" />


Supports PDF and common image formats (.png, .jpg, .jpeg).

Smooth panning and mouse-wheel zooming.

Zoom-under-mouse for natural navigation.

“Fit to View” button to quickly reset zoom.

🎯 Pick-on-Print Hotspots

Click-drag rectangles directly on the print to define hotspots.

Each hotspot is automatically assigned a unique ID (HS-001, HS-002, …).

Hotspots are persisted to CSV (blueprint.pdf.hotspots.csv).

Each hotspot stores:

Page, coordinates, width/height, zoom level

Inspection Method

Result value

Nominal, LSL, USL limits

🎈 Balloon Overlays

Each hotspot appears as a numbered red/white balloon overlay.

Balloons can be toggled on/off and dragged to reposition.

Positions are saved persistently per blueprint via QSettings.

Auto-highlight the selected balloon when a row is chosen.

📋 Table View & Filters

Editable table lists all hotspots.

Columns: ID, Page, Method, Result, Nominal, LSL, USL, Status

Color-coded rows:

✅ Green — PASS

❌ Red — FAIL

⚠️ Yellow — Indeterminate

Filter by Inspection Method or Status via combo boxes.

Auto-update status counts in the footer bar.

🔄 Ballooning vs. Inspection Mode
Mode	Purpose	Editable Columns
Ballooning	Engineering definition of hotspots & tolerances	All fields
Inspection	Enter per-serial inspection results	Result only

When loading a blueprint, you can choose the mode:

Ballooning: build and save the master hotspot plan.

Inspection: prompt for a work order / serial number, load saved hotspots, and overlay only the Result CSV.

🧠 Intelligent Auto-Fill from Result

Typing a tolerance expression in the Result field (Ballooning mode only) auto-fills the numeric fields:

1.000 ± .005 → Nominal = 1.000, LSL = 0.995, USL = 1.005

1.005+.005-.000 or 1.005-.000+.005 → Nominal = 1.005, LSL = 1.005, USL = 1.010
After filling, the Result cell is cleared automatically.

🧾 CSV Integration

Each blueprint maintains:

Global geometry CSV → blueprint.pdf.hotspots.csv

Per-work-order results CSV → blueprint.pdf.<workorder>.results.csv

Exports filtered views via “File → Export CSV…”.

🔍 OCR (Optical Character Recognition)

Run OCR on a selected hotspot to extract text or numbers directly from the print.

Tries native PDF text extraction first; falls back to pytesseract (Tesseract OCR).

Works with optional OpenCV preprocessing for improved accuracy.

OCR output is inserted into the Result cell and logged with confidence.

⚙️ Settings Persistence

Saves splitter sizes, window geometry, and balloon offsets using QSettings.

Remembers last used work order per blueprint.

🗑️ Editing Tools

Delete selected hotspots (keyboard Del or context menu).

Pick-mode toggle for adding new hotspots.

🧰 Tech Stack
Component	Purpose
Python 3.10+	Core language
PyQt6	GUI framework
PyMuPDF (fitz)	PDF rendering
OpenCV (cv2)	Image preprocessing for OCR
pytesseract	OCR engine
dataclasses / csv / QSettings	Data persistence
🪄 Installation
1️⃣ Clone & enter directory
git clone https://github.com/yourusername/inspection-app.git
cd inspection-app

2️⃣ Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt


If you don’t have a requirements.txt, create one with:

PyQt6
pymupdf
opencv-python-headless
pytesseract

4️⃣ Ensure Tesseract is installed

Linux:

sudo dnf install tesseract


or

sudo apt install tesseract-ocr


macOS:

brew install tesseract

🚀 Usage

Run directly:

python main.py


or (if using VS Code):

F5  →  Run main.py

First launch

File → Open Blueprint…
Choose a PDF or image blueprint.

Choose Ballooning or Inspection mode.

In Ballooning mode:

Enable “Pick” tool and drag to define hotspots.

Save or export anytime.

In Inspection mode:

Enter a work order number when prompted.

Input measured results for each hotspot.

Save results automatically per work order.

🧩 File Layout
inspection-app/
├── main.py                 # Application entry point
├── hotspots.csv            # Example dataset (optional)
├── blueprint.pdf           # Example blueprint (optional)
├── requirements.txt
└── README.md

🧠 Developer Notes
Hotspot CSV schema
Column	Type	Description
id	str	Unique ID (HS-001…)
page	int	Page number (0-based)
x, y, w, h	float	Coordinates & dimensions
zoom	float	Default zoom (%)
method	str	Inspection method (text)
result	str	Measured value or status
nominal	float	Target dimension
lsl / usl	float	Lower / upper spec limits
Work Order Results CSV
Column	Description
id	Matches hotspot ID
result	Measured value or pass/fail
Color coding

PASS → light green

FAIL → light red

— → light yellow

🧩 Recent Changes
🪶 Unequal Bilateral Tolerance Parser

Fixed regex crash (redefinition of group name 't') by renaming groups to plus and minus.

Now supports formats with spaces and Unicode symbols.

Strict re.fullmatch validation for cleaner parsing.

🪶 OCR Improvements

Auto-fallback from PDF text extraction → Tesseract OCR.

Adaptive threshold + morphology for improved accuracy.

🪶 Settings Enhancements

Added persistent window geometry and splitter sizes.

🔍 Known Limitations / Next Steps

No multi-page inspection export (one page at a time for now).

OCR accuracy depends heavily on print quality and font.

No database backend — currently file-based CSV storage.

Future roadmap:

🔸 Add search/filter by ID or text

🔸 Support units & additional tolerance syntaxes

🔸 Add keyboard navigation between balloons

🔸 Introduce image-based hotspot grouping or layers

🧑‍💻 Author

Brian Fosbinder
Midwest aerospace manufacturing engineer • CNC process optimization • Python / Power Apps integrator
📍 Quad Cities, IL

🪪 License

This project is released under the MIT License.
See LICENSE for details.
