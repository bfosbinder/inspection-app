import sys, os, csv, shutil
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF, pyqtSignal, QSettings, QSize
from PyQt6.QtGui import QAction, QPixmap, QImage, QPainter, QCursor, QFont, QPen, QBrush, QColor, QKeySequence, QIcon
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QTableWidget, QTableWidgetItem, QToolBar,
    QVBoxLayout, QFileDialog, QMessageBox, QLabel, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsRectItem, QHBoxLayout, QPushButton, QInputDialog,
    QStatusBar, QGraphicsEllipseItem, QGraphicsSimpleTextItem, QStyledItemDelegate, QComboBox, QLineEdit
)

# OCR / CV deps (optional until used)
try:
    import cv2  # opencv-python-headless
except ImportError:
    cv2 = None  # will alert user when OCR is invoked
try:
    import pytesseract
except ImportError:
    pytesseract = None
else:
    # Auto-detect system tesseract binary if available
    try:
        cmd = shutil.which("tesseract")
        if not cmd:
            for cand in ("/usr/bin/tesseract", "/usr/local/bin/tesseract"):
                if os.path.exists(cand):
                    cmd = cand; break
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
    except Exception:
        pass

# ---- PDF support (PyMuPDF) ----
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

# ---------- Data ----------
@dataclass
class Hotspot:
    id: str
    page: int
    x: float
    y: float
    w: float
    h: float
    zoom: float  # percent
    method: str = ""   # persisted
    result: str = ""   # persisted
    nominal: Optional[float] = None  # NEW: nominal/target value
    lsl: Optional[float] = None  # NEW: lower spec limit
    usl: Optional[float] = None  # NEW: upper spec limit

def load_hotspots(csv_path: str) -> List[Hotspot]:
    out: List[Hotspot] = []
    if not os.path.exists(csv_path):
        return out
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            # tolerant float parse
            def _f(val):
                try:
                    if val is None:
                        return None
                    s = str(val).strip()
                    if s == "":
                        return None
                    return float(s)
                except Exception:
                    return None
            out.append(
                Hotspot(
                    id=str(row.get("id", "")).strip(),
                    page=int(float(row.get("page", 0) or 0)),
                    x=float(row.get("x", 0) or 0),
                    y=float(row.get("y", 0) or 0),
                    w=float(row.get("w", 0) or 0),
                    h=float(row.get("h", 0) or 0),
                    zoom=float(row.get("zoom", 200) or 200),
                    method=str(row.get("method", "") or "").strip(),
                    result=str(row.get("result", "") or "").strip(),
                    nominal=_f(row.get("nominal")),
                    lsl=_f(row.get("lsl")),
                    usl=_f(row.get("usl")),
                )
            )
    return out

# ---------- Viewer with Pick-on-Print ----------
class ImageView(QGraphicsView):
    rectPicked = pyqtSignal(QRectF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        # zoom under mouse for nicer feel
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._pix_item: Optional[QGraphicsPixmapItem] = None
        self._highlight: Optional[QGraphicsRectItem] = None
        # pick mode state
        self._picking: bool = False
        self._rubber: Optional[QGraphicsRectItem] = None
        self._pick_start: Optional[QPointF] = None
        # balloons
        self._balloons: dict[str, QGraphicsEllipseItem] = {}
        self._balloon_size: float = 34.0  # bigger for readability
        # persistence context for balloon offsets (set by MainWindow)
        self._context_key: str = ""
        self._settings = QSettings("InspectionApp", "Balloons")

    def set_context_key(self, key: str):
        self._context_key = key or ""

    def set_pixmap(self, pm: QPixmap):
        self.scene().clear()
        # Reset references to any cleared items to avoid double-removal warnings
        self._highlight = None
        self._rubber = None
        self._balloons.clear()
        self._pix_item = self.scene().addPixmap(pm)
        self.scene().setSceneRect(self._pix_item.boundingRect())
        self.resetTransform()
        self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)

    def set_pick_mode(self, on: bool):
        self._picking = on
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor) if on else QCursor(Qt.CursorShape.ArrowCursor))
        self.setDragMode(QGraphicsView.DragMode.NoDrag if on else QGraphicsView.DragMode.ScrollHandDrag)
        # remove any temp rectangle
        if self._rubber:
            self.scene().removeItem(self._rubber)
            self._rubber = None
        self._pick_start = None

    def wheelEvent(self, e):  # smooth zoom on wheel
        if self._picking:
            e.ignore(); return
        if e.angleDelta().y() == 0:
            return
        factor = 1.15 if e.angleDelta().y() > 0 else 1/1.15
        self.scale(factor, factor)

    def mousePressEvent(self, e):
        if self._picking and e.button() == Qt.MouseButton.LeftButton:
            self._pick_start = self.mapToScene(e.pos())
            if self._rubber:
                self.scene().removeItem(self._rubber)
                self._rubber = None
            self._rubber = QGraphicsRectItem(QRectF(self._pick_start, self._pick_start))
            self._rubber.setPen(Qt.GlobalColor.red)
            self._rubber.setBrush(Qt.GlobalColor.yellow)
            self._rubber.setOpacity(0.25)
            self.scene().addItem(self._rubber)
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._picking and self._rubber and self._pick_start is not None:
            p = self.mapToScene(e.pos())
            rect = QRectF(self._pick_start, p).normalized()
            self._rubber.setRect(rect)
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._picking and e.button() == Qt.MouseButton.LeftButton and self._rubber:
            rect = self._rubber.rect()
            # guard tiny drags
            if rect.width() >= 2 and rect.height() >= 2:
                self.rectPicked.emit(rect)
            self.scene().removeItem(self._rubber)
            self._rubber = None
            self._pick_start = None
            return
        super().mouseReleaseEvent(e)

    def flash_rect(self, rect: QRectF, ms: int = 600):
        if self._highlight:
            self.scene().removeItem(self._highlight)
        self._highlight = QGraphicsRectItem(rect)
        self._highlight.setPen(Qt.GlobalColor.red)
        self._highlight.setBrush(QBrush(Qt.GlobalColor.yellow))
        self._highlight.setOpacity(0.35)
        self.scene().addItem(self._highlight)
        QTimer.singleShot(ms, self._clear_highlight)

    def _clear_highlight(self):
        # Safely remove highlight if it still exists in the scene
        if self._highlight and self._highlight.scene() is not None:
            self.scene().removeItem(self._highlight)
        self._highlight = None

    def center_on_rect(self, rect: QRectF, zoom_pct: float):
        self.centerOn(rect.center())
        self.resetTransform()
        vw = max(1, self.viewport().width())
        vh = max(1, self.viewport().height())
        if rect.width() == 0 or rect.height() == 0:
            scale_guess = 1.0
        else:
            sx = (vw * 0.4) / rect.width()
            sy = (vh * 0.4) / rect.height()
            scale_guess = min(sx, sy)
        if zoom_pct and zoom_pct > 0:
            scale_guess = zoom_pct / 100.0
        self.scale(scale_guess, scale_guess)
        self.centerOn(rect.center())
        self.flash_rect(rect)

    def clear_balloons(self):
        for item in self._balloons.values():
            self.scene().removeItem(item)
        self._balloons.clear()

    def set_balloons(self, items: list[tuple[str, QRectF, str]]):
        """
        items: list of (hotspot_id, rect, label_text)
        """
        self.clear_balloons()
        size = self._balloon_size
        for hs_id, rect, label in items:
            # place near rect top-left
            base_pos = rect.topLeft() - QPointF(size * 0.6, size * 0.6)
            dx, dy = self._load_offset(hs_id)
            pos = base_pos + QPointF(dx, dy)
            ellipse = BalloonItem(hs_id, base_pos, self._save_offset)
            ellipse.setPen(QPen(Qt.GlobalColor.red, 3))
            ellipse.setBrush(QBrush(Qt.GlobalColor.white))
            ellipse.setZValue(10)
            ellipse.setPos(pos)
            ellipse.setToolTip(hs_id)
            # text centered inside ellipse
            text = QGraphicsSimpleTextItem(label, ellipse)
            f = QFont(); f.setBold(True); f.setPointSizeF(11)
            text.setFont(f)
            tr = text.boundingRect()
            text.setPos((size - tr.width()) / 2, (size - tr.height()) / 2 - 1)
            text.setBrush(QBrush(Qt.GlobalColor.black))
            self.scene().addItem(ellipse)
            self._balloons[hs_id] = ellipse

    def highlight_balloon(self, hs_id: str):
        for k, item in self._balloons.items():
            if k == hs_id:
                item.setPen(QPen(Qt.GlobalColor.red, 2))
                item.setZValue(20)
            else:
                item.setPen(QPen(Qt.GlobalColor.darkGray, 1))
                item.setZValue(10)

    # ---- balloon offset persistence ----
    def _key_for_id(self, hs_id: str) -> str:
        return f"{self._context_key}|{hs_id}" if self._context_key else hs_id

    def _load_offset(self, hs_id: str) -> tuple[float, float]:
        key = self._key_for_id(hs_id)
        try:
            val = self._settings.value(key, "0,0", str)
            if isinstance(val, str):
                parts = val.split(",")
                if len(parts) == 2:
                    return float(parts[0]), float(parts[1])
        except Exception:
            pass
        return (0.0, 0.0)

    def _save_offset(self, hs_id: str, offset: QPointF):
        try:
            key = self._key_for_id(hs_id)
            self._settings.setValue(key, f"{offset.x():g},{offset.y():g}")
        except Exception:
            pass

    def set_balloons_movable(self, enabled: bool):
        for item in self._balloons.values():
            if isinstance(item, BalloonItem):
                item.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable, bool(enabled))

class BalloonItem(QGraphicsEllipseItem):
    def __init__(self, hs_id: str, base_pos: QPointF, save_cb, *args, **kwargs):
        super().__init__(0, 0, 34.0, 34.0, *args, **kwargs)
        self.hs_id = hs_id
        self.base_pos = QPointF(base_pos)
        self._save_cb = save_cb
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

    def mousePressEvent(self, event):
        self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        super().mouseReleaseEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.GraphicsItemChange.ItemPositionHasChanged:
            # save offset relative to base_pos
            try:
                new_pos: QPointF = value
            except Exception:
                new_pos = self.pos()
            offset = new_pos - self.base_pos
            if callable(self._save_cb):
                self._save_cb(self.hs_id, offset)
        return super().itemChange(change, value)

# ---------- Delegates ----------
class MethodComboDelegate(QStyledItemDelegate):
    """Editable combo box that offers existing methods as suggestions."""
    def __init__(self, parent=None, get_values=None):
        super().__init__(parent)
        self._get_values = get_values  # callable -> list[str]

    def createEditor(self, parent, option, index):
        cb = QComboBox(parent)
        cb.setEditable(True)
        # populate with known values
        if callable(self._get_values):
            values = [v for v in self._get_values() if v]
            seen = set()
            for v in values:
                if v not in seen:
                    cb.addItem(v)
                    seen.add(v)
        return cb

    def setEditorData(self, editor, index):
        if isinstance(editor, QComboBox):
            text = index.data() or ""
            editor.setCurrentText(text)
        else:
            super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        if isinstance(editor, QComboBox):
            model.setData(index, editor.currentText())
        else:
            super().setModelData(editor, model, index)

# ---------- Blueprint provider ----------
class Blueprint:
    def __init__(self, path: str, scale: float = 2.0):
        self.path = path
        self.scale = scale
        self._doc = None
        self._img_pixmap: Optional[QPixmap] = None
        self._cache: dict[int, QPixmap] = {}
        if self.is_pdf and fitz is None:
            raise RuntimeError("PyMuPDF (pymupdf) is not installed. `pip install pymupdf`")
        if self.is_pdf:
            self._doc = fitz.open(self.path)

    @property
    def is_pdf(self) -> bool:
        return self.path.lower().endswith(".pdf")

    def page_count(self) -> int:
        return self._doc.page_count if self._doc else 1

    def render_page(self, page_index: int) -> QPixmap:
        if not self.is_pdf:
            if self._img_pixmap is None:
                img = QImage(self.path)
                if img.isNull():
                    raise RuntimeError(f"Failed to load image: {self.path}")
                self._img_pixmap = QPixmap.fromImage(img)
            return self._img_pixmap
        if page_index in self._cache:
            return self._cache[page_index]
        page = self._doc.load_page(page_index)
        mat = fitz.Matrix(self.scale, self.scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format.Format_RGB888)
        pm = QPixmap.fromImage(img.copy())
        self._cache[page_index] = pm
        return pm

# ---------- Main Window ----------
class MainWindow(QMainWindow):
    def __init__(self, blueprint_path: str = "blueprint.pdf", csv_path: str = "hotspots.csv"):
        super().__init__()
        self._title_base = "Inspection Plan (Python) — Pick-on-Print mode"
        self.setWindowTitle(self._title_base)
        self.resize(1200, 800)

        self.status = QStatusBar(); self.setStatusBar(self.status)
        # Start blank: no auto-loaded blueprint or hotspots
        self.blueprint_path = ""
        # Global geometry/methods/limits CSV (per blueprint)
        self.hotspots_path_global = ""
        # Per-serial results CSV (overlay; only id/result)
        self.results_path_wo = ""
        self.work_order: str = ""
        # mode: 'ballooning' (full edit) or 'inspection' (only results editable)
        self.mode: str = "ballooning"
        self.bp: Optional[Blueprint] = None
        self.hotspots: List[Hotspot] = []
        self.current_page: int = 0
        # Guard to suppress any auto-saves during load/open flows
        self._is_loading: bool = False

        # UI
        self.view = ImageView()
        self.view.rectPicked.connect(self._picked_rect)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["ID", "Page", "Inspection Method", "Result", "Nominal", "LSL", "USL", "Status"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.DoubleClicked | QTableWidget.EditTrigger.EditKeyPressed)
        self._updating_table = False
        # Editable combo delegate for "Inspection Method"
        self.method_delegate = MethodComboDelegate(self.table, self._collect_methods)
        self.table.setItemDelegateForColumn(2, self.method_delegate)
        self.table.itemChanged.connect(self._on_item_changed)

        # Apply readability settings (fonts, padding, colors)
        self._apply_readability()

        left_box = QWidget(); left_layout = QVBoxLayout(left_box)
        # Top button bar (capped height so it doesn't stretch vertically)
        btns_widget = QWidget()
        btns = QHBoxLayout(btns_widget)
        # Small toolbar with Fit and Pick actions to save vertical space
        tools = QToolBar("Tools", self)
        tools.setIconSize(QSize(18, 18))
        tools.setMovable(False)
        tools.setFloatable(False)
        tools.setStyleSheet("QToolBar { border: 0; }")
        # Actions
        self.act_fit = QAction(QIcon.fromTheme("zoom-fit-best"), "Fit", self)
        self.act_fit.setToolTip("Fit the blueprint to view")
        self.act_pick = QAction(QIcon.fromTheme("selection-rectangular"), "Pick", self)
        self.act_pick.setToolTip("Pick Hotspot: click-drag a rectangle on the print")
        self.act_pick.setCheckable(True)
        tools.addAction(self.act_fit)
        tools.addAction(self.act_pick)
        self.btn_balloons = QPushButton("Balloons")
        self.btn_balloons.setCheckable(True)
        self.btn_balloons.setChecked(True)
        # OCR button
        self.btn_ocr = QPushButton("OCR Result")
        self.btn_ocr.setToolTip("Run OCR on selected hotspot")
        # Lay out buttons
        btns.addWidget(tools)
        btns.addWidget(self.btn_balloons)
        btns.addWidget(self.btn_ocr)
        # Filter by Inspection Method
        self.lbl_filter = QLabel("Filter:")
        self.filter_method = QComboBox()
        self.filter_method.setEditable(False)
        self.filter_method.addItem("(All)")
        self.filter_method.currentIndexChanged.connect(self._on_filter_method_changed)
        # Filter by Status
        self.lbl_status_filter = QLabel("Status:")
        self.filter_status = QComboBox()
        self.filter_status.setEditable(False)
        self.filter_status.addItems(["(Any status)", "PASS", "FAIL", "—"])
        self.filter_status.currentIndexChanged.connect(self._on_filter_status_changed)
        btns.addSpacing(12)
        btns.addWidget(self.lbl_filter)
        btns.addWidget(self.filter_method)
        btns.addSpacing(8)
        btns.addWidget(self.lbl_status_filter)
        btns.addWidget(self.filter_status)
        btns.addStretch(1)
        # Cap the button bar height to keep left panel compact
        try:
            btns_widget.setMaximumHeight(40)
        except Exception:
            pass
        left_layout.addWidget(btns_widget)
        left_layout.addWidget(self.table)

        # Splitter: make right side (preview) larger by default and remember size
        sp = QSplitter()
        sp.addWidget(left_box)
        sp.addWidget(self.view)
        # Prefer right pane 2–3× wider than left by default
        sp.setStretchFactor(0, 1)
        sp.setStretchFactor(1, 3)
        sp.setSizes([400, 1000])
        self.splitter = sp
        self.setCentralWidget(sp)
        # Restore previous splitter sizes if available
        self.settings = QSettings("InspectionApp", "MainWindow")
        try:
            sizes = self.settings.value("splitterSizes")
            if sizes:
                sp.setSizes([int(s) for s in sizes])
        except Exception:
            pass

        # menu
        file_menu = self.menuBar().addMenu("&File")
        act_open_bp = QAction("Open Blueprint…", self); act_open_bp.triggered.connect(self._choose_blueprint)
        act_open_csv = QAction("Open Hotspots/Results…", self); act_open_csv.triggered.connect(self._choose_hotspots)
        act_export = QAction("Export CSV…", self); act_export.triggered.connect(self._export_csv)
        file_menu.addAction(act_open_bp); file_menu.addAction(act_open_csv); file_menu.addAction(act_export)

        # signals
        self.table.selectionModel().selectionChanged.connect(self._row_selected)
        self.act_fit.triggered.connect(lambda: self.view.fitInView(self.view.scene().itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio))
        self.act_pick.toggled.connect(self._toggle_pick)
        self.btn_balloons.toggled.connect(self._toggle_balloons)
        self.btn_ocr.clicked.connect(self._ocr_selected_row)
        self._balloons_on = True

        # Table delete action (keyboard shortcut + context menu)
        self.act_delete = QAction("Delete Row", self)
        self.act_delete.setShortcut(QKeySequence.StandardKey.Delete)
        self.act_delete.triggered.connect(self._delete_selected_rows)
        self.table.addAction(self.act_delete)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)

        # start with blank preview and empty checklist
        self.status.showMessage("Ready. Open a blueprint and hotspots from the File menu to begin.")

    # ---- mode handling ----
    def _set_mode(self, mode: str):
        mode = (mode or "").strip().lower()
        if mode not in ("ballooning", "inspection"):
            mode = "ballooning"
        self.mode = mode
        # enable/disable actions according to mode
        if hasattr(self, "act_pick"):
            self.act_pick.setEnabled(self.mode == "ballooning")
            if self.mode != "ballooning" and self.act_pick.isChecked():
                self.act_pick.setChecked(False)
        if hasattr(self, "act_delete"):
            self.act_delete.setEnabled(self.mode == "ballooning")
        # OCR allowed in inspection (assists entering results); keep enabled
        # Show/hide OCR button based on mode (don't show in Inspection mode)
        if hasattr(self, "btn_ocr"):
            try:
                self.btn_ocr.setVisible(self.mode != "inspection")
            except Exception:
                pass
        # Toggle balloon drag capability
        if hasattr(self, "view") and hasattr(self.view, "_balloons"):
            try:
                self.view.set_balloons_movable(self.mode == "ballooning")
            except Exception:
                pass
        # Apply table cell edit permissions
        self._apply_table_permissions()

    def _apply_table_permissions(self):
        rows = self.table.rowCount()
        cols = self.table.columnCount()
        for r in range(rows):
            for c in range(cols):
                it = self.table.item(r, c)
                if it is None:
                    it = QTableWidgetItem("")
                    self.table.setItem(r, c, it)
                flags = it.flags()
                # Status column is always read-only
                editable = (c == 3) if self.mode == "inspection" else (c != 7)
                if editable:
                    it.setFlags(flags | Qt.ItemFlag.ItemIsEditable)
                else:
                    it.setFlags(flags & ~Qt.ItemFlag.ItemIsEditable)

    # ---- helpers ----
    def _csv_path_for_blueprint(self, bp_path: str) -> str:
        # Global sidecar CSV next to the blueprint file (shared across serials)
        return f"{bp_path}.hotspots.csv"

    def _results_path_for_blueprint(self, bp_path: str, work_order: str) -> str:
        # Per-serial results sidecar
        safe = self._sanitize_work_order(work_order or "")
        return f"{bp_path}.{safe}.results.csv"

    def _find_blueprint(self, default: str) -> str:
        if os.path.exists(default): return default
        for alt in ("blueprint.png", "blueprint.jpg", "blueprint.jpeg"):
            if os.path.exists(alt): return alt
        return default

    def _apply_readability(self):
        # Base application font bump
        app = QApplication.instance()
        base = app.font() if app else self.font()
        pt = base.pointSizeF() if base.pointSizeF() > 0 else 10.0
        base.setPointSizeF(max(11.0, pt + 2))
        if app:
            app.setFont(base)

        # Table font and headers
        table_font = QFont(base)
        table_font.setPointSizeF(base.pointSizeF())
        self.table.setFont(table_font)
        header_font = QFont(table_font); header_font.setBold(True)
        hh = self.table.horizontalHeader(); vh = self.table.verticalHeader()
        hh.setFont(header_font); vh.setFont(header_font)
        # Row height and header sizes
        row_h = max(28, self.table.fontMetrics().height() + 12)
        vh.setDefaultSectionSize(row_h)
        hh.setStretchLastSection(True)
        # Alternating rows and padding + softer selection overlay
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            """
            QTableWidget { gridline-color: #888; }
            QTableWidget::item { padding: 6px; }
            QTableView::item:selected { background: rgba(30,144,255,160); color: black; }
            QHeaderView::section { padding: 6px; font-weight: bold; }
            """
        )

    def _update_window_title(self):
        parts: list[str] = []
        if self.blueprint_path:
            parts.append(os.path.basename(self.blueprint_path))
        if self.work_order and self.mode == "inspection":
            parts.append(f"WO {self.work_order}")
        if parts:
            self.setWindowTitle(f"{self._title_base} — {' · '.join(parts)}")
        else:
            self.setWindowTitle(self._title_base)

    def _work_order_settings_key(self, blueprint_path: str) -> str:
        return f"workOrders/{os.path.abspath(blueprint_path)}"

    def _sanitize_work_order(self, work_order: str) -> str:
        import re
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", work_order.strip())
        return safe or "workorder"

    def _prompt_work_order(self, blueprint_path: str) -> Optional[str]:
        last = ""
        if hasattr(self, "settings"):
            key = self._work_order_settings_key(blueprint_path)
            last = self.settings.value(key, "", str) or ""
        text = last
        while True:
            text, ok = QInputDialog.getText(
                self,
                "Select Work Order",
                f"Enter work order (serial) for\n{os.path.basename(blueprint_path)}:",
                QLineEdit.EchoMode.Normal,
                text,
            )
            if not ok:
                return None
            text = text.strip()
            if text:
                if hasattr(self, "settings"):
                    self.settings.setValue(self._work_order_settings_key(blueprint_path), text)
                return text
            QMessageBox.information(self, "Work Order Required", "Please enter a work order number or press Cancel.")

    def _infer_work_order_from_csv(self, csv_path: str, blueprint_path: Optional[str] = None) -> Optional[str]:
        name = os.path.basename(csv_path)
        suffixes = [".hotspots.csv", ".results.csv"]
        use_suffix = None
        for s in suffixes:
            if name.endswith(s):
                use_suffix = s
                break
        if not use_suffix:
            return None
        stem = name[: -len(use_suffix)]
        if blueprint_path:
            bp_name = os.path.basename(blueprint_path)
            prefix = f"{bp_name}."
            if stem.startswith(prefix):
                inferred = stem[len(prefix):]
                return inferred or ""
            if stem == bp_name:
                return ""
        # fall back to last dotted segment
        if "." in stem:
            return stem.rsplit(".", 1)[-1]
        return None

    def _update_status_counts(self):
        rows = self.table.rowCount()
        c_pass = c_fail = c_ind = 0
        for r in range(rows):
            if self.table.isRowHidden(r):
                continue
            it = self.table.item(r, 7)
            val = (it.text().strip() if it else "—")
            if val == "PASS":
                c_pass += 1
            elif val == "FAIL":
                c_fail += 1
            else:
                c_ind += 1
        self.status.showMessage(f"PASS {c_pass} · FAIL {c_fail} · — {c_ind}")

    def _fmt_num(self, v: Optional[float]) -> str:
        return "" if v is None else f"{v:g}"

    def _load_blueprint(self, path: str):
        if not os.path.exists(path):
            QMessageBox.information(self, "Blueprint", "Place 'blueprint.pdf' (or .png/.jpg) here, or click 'Open Blueprint'.")
            return
        try:
            self.bp = Blueprint(path, scale=2.0)
            self.current_page = 0
            pm = self.bp.render_page(self.current_page)
            self.view.set_pixmap(pm)
            suffix = f" · WO {self.work_order}" if (self.work_order and self.mode == "inspection") else ""
            self.status.showMessage(f"Loaded blueprint: {os.path.basename(path)}{suffix}")
            self._update_window_title()
            # set context for balloon persistence (global across serials)
            if hasattr(self.view, "set_context_key"):
                self.view.set_context_key(self._balloon_context_key())
            if self._balloons_on:
                self._refresh_balloons_for_page()
        except Exception as e:
            QMessageBox.critical(self, "Blueprint error", str(e))

    def _load_hotspots(self, path: str):
        # create sample if missing (with method/result/lsl/usl columns)
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["id","page","x","y","w","h","zoom","method","result","nominal","lsl","usl"])
                # Global starter examples if you want; omit when using work orders
                if not self.work_order:
                    w.writerow(["DIM-A1",0,1234,880,280,180,220,"","", "", "", ""])
                    w.writerow(["DIM-B7",0,1880,1420,260,160,220,"","", "", "", ""])
        try:
            self.hotspots = load_hotspots(path)
        except Exception as e:
            QMessageBox.critical(self, "Hotspots error", str(e)); self.hotspots = []
        self._refresh_table()

    def _load_results_overlay(self, results_path: str):
        """Overlay per-serial results (id -> result) onto current hotspots."""
        if not results_path or not os.path.exists(results_path):
            return
        try:
            with open(results_path, newline="", encoding="utf-8-sig") as f:
                r = csv.DictReader(f)
                # accept both minimal (id,result) or full schema with 'result'
                by_id: dict[str, str] = {}
                for row in r:
                    rid = str(row.get("id", "")).strip()
                    res = str(row.get("result", "")).strip()
                    # Only apply non-empty results
                    if rid and res != "":
                        by_id[rid] = res
            # Apply overlay and clear results for IDs not present in this serial's CSV
            present = set(by_id.keys())
            applied = 0
            cleared = 0
            for hs in self.hotspots:
                if hs.id in present:
                    hs.result = by_id[hs.id]
                    applied += 1
                else:
                    if hs.result:
                        cleared += 1
                    hs.result = ""
            try:
                self.status.showMessage(f"Loaded results: applied {applied}, cleared {cleared}")
            except Exception:
                pass
        except Exception as e:
            QMessageBox.critical(self, "Results error", str(e))
    
    def _clear_results(self):
        """Clear all Result values in memory (used when switching serials)."""
        for hs in self.hotspots:
            hs.result = ""

    def _refresh_table(self):
        self._updating_table = True
        try:
            self.table.setRowCount(len(self.hotspots))
            for i, hs in enumerate(self.hotspots):
                self.table.setItem(i, 0, QTableWidgetItem(hs.id))
                self.table.setItem(i, 1, QTableWidgetItem(str(hs.page)))
                self.table.setItem(i, 2, QTableWidgetItem(hs.method or ""))
                self.table.setItem(i, 3, QTableWidgetItem(hs.result or ""))
                self.table.setItem(i, 4, QTableWidgetItem(self._fmt_num(hs.nominal)))
                self.table.setItem(i, 5, QTableWidgetItem(self._fmt_num(hs.lsl)))
                self.table.setItem(i, 6, QTableWidgetItem(self._fmt_num(hs.usl)))
                status_item = QTableWidgetItem(self._compute_status(hs))
                status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(i, 7, status_item)
                self._paint_row_status(i)
            self.table.resizeColumnsToContents()
        finally:
            self._updating_table = False
        self._update_method_filter_options()
        self._apply_filters()
        self._apply_table_permissions()

    def _choose_blueprint(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open blueprint", "", "PDF or Images (*.pdf *.png *.jpg *.jpeg)")
        if not path:
            return
        self._is_loading = True
        try:
            # Choose mode for this session first (default to Inspection for PDFs)
            default_idx = 1 if str(path).lower().endswith(".pdf") else 0
            mode, ok = QInputDialog.getItem(
                self,
                "Select Mode",
                "Choose how you want to use this blueprint:",
                ["Ballooning", "Inspection"],
                default_idx,
                False,
            )
            if not ok:
                return
            self._set_mode("ballooning" if mode.lower().startswith("balloon") else "inspection")
            # Prompt for serial only in Inspection mode
            work_order = ""
            if self.mode == "inspection":
                work_order = self._prompt_work_order(path) or ""
                if not work_order:
                    return
            self.blueprint_path = path
            self.work_order = work_order if self.mode == "inspection" else ""
            # Paths
            self.hotspots_path_global = self._csv_path_for_blueprint(path)
            self.results_path_wo = self._results_path_for_blueprint(path, work_order) if self.mode == "inspection" else ""
            # Load blueprint and data
            self._load_blueprint(path)
            self._load_hotspots(self.hotspots_path_global)
            if self.mode == "inspection":
                # For a known serial (existing results CSV), don't clear in-memory results
                # so we preserve previously entered values; otherwise start clean.
                should_clear = not (self.results_path_wo and os.path.exists(self.results_path_wo))
                if should_clear:
                    self._clear_results()
                self._load_results_overlay(self.results_path_wo)
                self._refresh_table()
            self._apply_filters()
            self._update_window_title()
            if hasattr(self.view, "set_context_key"):
                self.view.set_context_key(self._balloon_context_key())
        finally:
            self._is_loading = False

    def _choose_hotspots(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Hotspots/Results CSV", "", "CSV (*.csv)")
        if not path:
            return
        self._is_loading = True
        try:
            # Determine how to use selection based on mode
            if self.mode == "ballooning":
                self.hotspots_path_global = path
                self._load_hotspots(path)
            else:
                # Inspection: treat as results overlay
                inferred = self._infer_work_order_from_csv(path, self.blueprint_path if self.blueprint_path else None)
                if inferred is not None:
                    self.work_order = inferred
                self.results_path_wo = path
                # Reload global geometry if needed
                if self.hotspots_path_global and os.path.exists(self.hotspots_path_global):
                    self._load_hotspots(self.hotspots_path_global)
                if self.results_path_wo:
                    self._load_results_overlay(self.results_path_wo)
                self._refresh_table()
            self._apply_filters()
            self._update_window_title()
            if hasattr(self.view, "set_context_key"):
                self.view.set_context_key(self._balloon_context_key())
        finally:
            self._is_loading = False

    def _show_page(self, page_index: int):
        if not self.bp: return
        page_index = max(0, page_index)
        self.current_page = page_index
        pm = self.bp.render_page(page_index)
        self.view.set_pixmap(pm)
        if self._balloons_on:
            self._refresh_balloons_for_page()
        self.status.showMessage(f"Page {page_index}")

    def _refresh_balloons_for_page(self):
        page_hs = [hs for hs in self.hotspots if hs.page == self.current_page]
        items: list[tuple[str, QRectF, str]] = []
        for idx, hs in enumerate(page_hs, start=1):
            rect = QRectF(hs.x, hs.y, max(2, hs.w), max(2, hs.h))
            items.append((hs.id, rect, str(idx)))
        self.view.set_balloons(items)
        self._update_status_counts()

    def _balloon_context_key(self) -> str:
        # Globalize offsets: same positions across serials
        return os.path.abspath(self.blueprint_path) if self.blueprint_path else ""

    def _toggle_balloons(self, on: bool):
        self._balloons_on = on
        if on:
            self._refresh_balloons_for_page()
            self.status.showMessage("Balloons on")
        else:
            self.view.clear_balloons()
            self.status.showMessage("Balloons off")

    def _collect_methods(self) -> list[str]:
        methods: list[str] = []
        seen: set[str] = set()
        rows = self.table.rowCount()
        for r in range(rows):
            it = self.table.item(r, 2)
            val = (it.text().strip() if it else "")
            if val and val not in seen:
                methods.append(val)
                seen.add(val)
        return methods

    def _update_method_filter_options(self):
        rows = self.table.rowCount()
        has_empty = False
        for r in range(rows):
            it = self.table.item(r, 2)
            if it is None or it.text().strip() == "":
                has_empty = True
                break
        existing = self.filter_method.currentText() if hasattr(self, 'filter_method') else "(All)"
        values = self._collect_methods()
        if hasattr(self, 'filter_method'):
            self.filter_method.blockSignals(True)
            self.filter_method.clear()
            self.filter_method.addItem("(All)")
            if has_empty:
                self.filter_method.addItem("(No method)")
            for v in values:
                self.filter_method.addItem(v)
            idx = self.filter_method.findText(existing)
            self.filter_method.setCurrentIndex(0 if idx < 0 else idx)
            self.filter_method.blockSignals(False)

    def _apply_filters(self):
        if not hasattr(self, 'filter_method') or not hasattr(self, 'filter_status'):
            return
        sel_method = self.filter_method.currentText()
        sel_status = self.filter_status.currentText()
        rows = self.table.rowCount()
        for r in range(rows):
            itm = self.table.item(r, 2)
            method_val = (itm.text().strip() if itm else "")
            method_ok = True
            if sel_method == "(All)":
                method_ok = True
            elif sel_method == "(No method)":
                method_ok = (method_val == "")
            else:
                method_ok = (method_val == sel_method)
            its = self.table.item(r, 7)
            status_val = (its.text().strip() if its else "—")
            status_ok = True
            if sel_status == "(Any status)":
                status_ok = True
            else:
                status_ok = (status_val == sel_status)
            self.table.setRowHidden(r, not (method_ok and status_ok))
        self._update_status_counts()

    def _on_filter_method_changed(self):
        self._apply_filters()

    def _on_filter_status_changed(self):
        self._apply_filters()

    # ---- numeric helpers & row paint ----
    def _parse_value(self, text: str) -> Optional[float]:
        import re
        if text is None:
            return None
        s = str(text).strip()
        if s == "":
            return None
        s = s.replace("Ø", "")
        m = re.search(r"[-+]?((?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][-+]?\d+)?", s)
        if not m:
            return None
        try:
            return float(m.group(0))
        except Exception:
            return None

    def _parse_nominal_tol(self, text: str) -> Optional[tuple[float, float]]:
        import re
        if not text:
            return None
        s = str(text).strip().replace("Ø", "")
        s = s.replace("+/-", "±").replace("+/−", "±").replace("＋/－", "±")
        m = re.search(r"([-+]?((?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][-+]?\d+)?)\s*±\s*(((?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][-+]?\d+)?)", s)
        if not m:
            return None
        try:
            nom = float(m.group(1))
            tol = float(m.group(3))
            return (nom, tol)
        except Exception:
            return None

    def _parse_nominal_unequal_tol(self, text: str) -> Optional[tuple[float, float, float]]:
        """Parse unequal bilateral tolerance like '1.005+.005-.000' or '1.005-.000+.005'.
        Returns (nominal, plus_tol, minus_tol) if matched, else None.
        - Uses fullmatch to avoid partial matches
        - Normalizes Unicode signs and strips 'Ø'
        - Ignores trailing tokens like TYP, REF, units (in, mm)
        - Allows optional whitespace between signs and numbers
        """
        import re
        if not text:
            return None
        s = str(text).strip()
        # Normalize symbols and stray tokens
        s = (s.replace("Ø", "")
               .replace("＋", "+").replace("－", "-").replace("−", "-")
             )
        # Drop annotations like TYP/REF and trailing units
        s = re.sub(r"\b(TYP|REF)\b.*$", "", s, flags=re.I).strip()
        s = re.sub(r"\b(?:in|mm)\b$", "", s, flags=re.I).strip()

        num = r"(?P<nom>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
        tol = r"(?P<t>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
        plus  = rf"\+\s*{tol}"
        minus = rf"-\s*{tol}"

        # Case 1: nominal + plus_tol - minus_tol
        m = re.fullmatch(rf"{num}\s*{plus}\s*{minus}", s)
        if m:
            try:
                nom = float(m.group('nom'))
                vals = [float(x) for x in re.findall(rf"{tol}", m.group(0))]
                if len(vals) == 2:
                    plus_tol, minus_tol = vals[0], vals[1]
                    return (nom, plus_tol, minus_tol)
            except Exception:
                return None

        # Case 2: nominal - minus_tol + plus_tol
        m = re.fullmatch(rf"{num}\s*{minus}\s*{plus}", s)
        if m:
            try:
                nom = float(m.group('nom'))
                vals = [float(x) for x in re.findall(rf"{tol}", m.group(0))]
                if len(vals) == 2:
                    minus_tol, plus_tol = vals[0], vals[1]
                    return (nom, plus_tol, minus_tol)
            except Exception:
                return None
        return None

    def _num_or_none(self, text: str) -> Optional[float]:
        return self._parse_value(text)

    def _judge(self, value: Optional[float], lsl: Optional[float], usl: Optional[float]) -> str:
        if value is None or (lsl is None and usl is None):
            return "ind"
        if lsl is not None and value < lsl:
            return "fail"
        if usl is not None and value > usl:
            return "fail"
        return "pass"

    def _compute_status(self, hs: Hotspot) -> str:
        t = (hs.result or "").strip().lower()
        if t in ("p", "pass"):
            return "PASS"
        if t in ("f", "fail"):
            return "FAIL"
        value = self._parse_value(hs.result)
        verdict = self._judge(value, hs.lsl, hs.usl)
        if verdict == "pass":
            return "PASS"
        if verdict == "fail":
            return "FAIL"
        return "—"

    def _paint_row_status(self, row: int):
        if row < 0 or row >= len(self.hotspots):
            return
        hs = self.hotspots[row]
        status_text = self._compute_status(hs)
        status = "pass" if status_text == "PASS" else ("fail" if status_text == "FAIL" else "ind")
        if status == "pass":
            color = QColor(170, 255, 170)
        elif status == "fail":
            color = QColor(255, 170, 170)
        else:
            color = QColor(255, 235, 150)
        cols = self.table.columnCount()
        for c in range(cols):
            item = self.table.item(row, c)
            if item is None:
                item = QTableWidgetItem("")
                self.table.setItem(row, c, item)
            item.setBackground(QBrush(color))
            item.setForeground(QBrush(Qt.GlobalColor.black))
        it = self.table.item(row, 7)
        if it is None:
            it = QTableWidgetItem("")
            self.table.setItem(row, 7, it)
        it.setText(status_text)
        it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)

    def _on_item_changed(self, item: QTableWidgetItem):
        if getattr(self, "_updating_table", False) or getattr(self, "_is_loading", False):
            return
        row = item.row()
        if row < 0 or row >= len(self.hotspots):
            return
        col = item.column()
        text = (item.text() or "").strip()
        if col == 2:  # Inspection Method
            self.hotspots[row].method = text
            if self.mode == "ballooning":
                self._save_hotspots_csv()
            self._update_method_filter_options()
            self._apply_filters()
        elif col == 3:  # Result
            hs = self.hotspots[row]
            hs.result = text
            nom_tol = self._parse_nominal_tol(text)
            if nom_tol and self.mode == "ballooning":
                nom, tol = nom_tol
                if hs.lsl is None and hs.usl is None:
                    if hs.nominal is None:
                        hs.nominal = nom
                    if hs.lsl is None:
                        hs.lsl = nom - tol
                    if hs.usl is None:
                        hs.usl = nom + tol
                    self._updating_table = True
                    try:
                        self.table.item(row, 4).setText(self._fmt_num(hs.nominal))
                        self.table.item(row, 5).setText(self._fmt_num(hs.lsl))
                        self.table.item(row, 6).setText(self._fmt_num(hs.usl))
                        hs.result = ""
                        self.table.item(row, 3).setText("")
                    finally:
                        self._updating_table = False
            else:
                # Try unequal bilateral tolerance: e.g., 1.005+.005-.000
                uneq = self._parse_nominal_unequal_tol(text)
                if uneq and self.mode == "ballooning":
                    nom, plus_tol, minus_tol = uneq
                    if hs.lsl is None and hs.usl is None:
                        if hs.nominal is None:
                            hs.nominal = nom
                        if hs.lsl is None:
                            hs.lsl = nom - minus_tol
                        if hs.usl is None:
                            hs.usl = nom + plus_tol
                        self._updating_table = True
                        try:
                            self.table.item(row, 4).setText(self._fmt_num(hs.nominal))
                            self.table.item(row, 5).setText(self._fmt_num(hs.lsl))
                            self.table.item(row, 6).setText(self._fmt_num(hs.usl))
                            hs.result = ""
                            self.table.item(row, 3).setText("")
                        finally:
                            self._updating_table = False
            if self.mode == "inspection":
                self._save_results_csv()
            else:
                self._save_hotspots_csv()
            self._paint_row_status(row)
            self._apply_filters()
        elif col == 4:  # Nominal
            self.hotspots[row].nominal = self._parse_value(text)
            if self.mode == "ballooning":
                self._save_hotspots_csv()
            self._paint_row_status(row)
            self._apply_filters()
        elif col == 5:  # LSL
            self.hotspots[row].lsl = self._parse_value(text)
            if self.mode == "ballooning":
                self._save_hotspots_csv()
            self._paint_row_status(row)
            self._apply_filters()
        elif col == 6:  # USL
            self.hotspots[row].usl = self._parse_value(text)
            if self.mode == "ballooning":
                self._save_hotspots_csv()
            self._paint_row_status(row)
            self._apply_filters()

    # ---- OCR helpers ----
    def _qimage_to_bgr(self, qimg: QImage):
        import numpy as np
        qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
        w, h = qimg.width(), qimg.height()
        if w <= 0 or h <= 0:
            return np.zeros((0, 0, 3), dtype=np.uint8)
        bytes_per_line = qimg.bytesPerLine()
        ptr = qimg.bits()
        arr = None
        total = int(bytes_per_line) * int(h)
        try:
            ptr.setsize(total)
            arr = np.frombuffer(ptr, dtype=np.uint8, count=total).reshape((h, bytes_per_line))
        except Exception:
            try:
                buf = ptr.asstring(total)
                arr = np.frombuffer(buf, dtype=np.uint8, count=total).reshape((h, bytes_per_line))
            except Exception:
                qimg2 = qimg.copy()
                ptr2 = qimg2.bits()
                try:
                    ptr2.setsize(total)
                    arr = np.frombuffer(ptr2, dtype=np.uint8, count=total).reshape((h, bytes_per_line))
                except Exception:
                    return np.zeros((0, 0, 3), dtype=np.uint8)
        rgb = arr[:, : w * 3].reshape((h, w, 3))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if cv2 is not None else rgb[:, :, ::-1]

    def _render_hotspot_clip(self, hs: Hotspot, dpi: int = 300):
        import numpy as np
        if self.bp and self.bp.is_pdf and fitz is not None and getattr(self.bp, "_doc", None) is not None:
            try:
                page = self.bp._doc.load_page(int(hs.page))
                scale_display = float(getattr(self.bp, "scale", 2.0) or 2.0)
                rect_pts = fitz.Rect(
                    float(hs.x) / scale_display,
                    float(hs.y) / scale_display,
                    (float(hs.x) + float(max(1.0, hs.w))) / scale_display,
                    (float(hs.y) + float(max(1.0, hs.h))) / scale_display,
                )
                mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
                pix = page.get_pixmap(matrix=mat, clip=rect_pts, alpha=False)
                qimg = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format.Format_RGB888)
                qimg = qimg.copy()
                return self._qimage_to_bgr(qimg)
            except Exception:
                pass
        if self.view and getattr(self.view, "_pix_item", None) is not None and self.view._pix_item.pixmap() and not self.view._pix_item.pixmap().isNull():
            pm = self.view._pix_item.pixmap()
            img_w, img_h = pm.width(), pm.height()
            x = int(max(0, min(img_w - 1, round(hs.x))))
            y = int(max(0, min(img_h - 1, round(hs.y))))
            w = int(max(1, round(hs.w)))
            h = int(max(1, round(hs.h)))
            if x + w > img_w:
                w = max(1, img_w - x)
            if y + h > img_h:
                h = max(1, img_h - y)
            sub = pm.copy(x, y, w, h)
            return self._qimage_to_bgr(sub.toImage())
        return np.zeros((0, 0, 3), dtype=np.uint8)

    def _prep_for_ocr(self, bgr):
        import numpy as np
        if bgr is None or bgr.size == 0:
            return bgr
        if cv2 is None:
            return bgr
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
        if np.mean(thr) < 127:
            thr = cv2.bitwise_not(thr)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
        return closed

    def _ocr_text(self, img: "np.ndarray", digits_only: bool = True) -> tuple[str, float]:
        if pytesseract is None:
            raise RuntimeError("pytesseract is not installed. Install it to use OCR.")
        cfg = "--psm 7"
        if digits_only:
            cfg += " -c tessedit_char_whitelist=0123456789.-+eE"
        data = pytesseract.image_to_data(img, lang="eng", config=cfg, output_type=pytesseract.Output.DICT)
        words = []
        confs = []
        n = len(data.get("text", []))
        for i in range(n):
            t = (data["text"][i] or "").strip()
            c = data.get("conf", ["-1"])[i]
            try:
                cval = float(c)
            except Exception:
                cval = -1.0
            if t and cval >= 0:
                words.append(t)
                confs.append(cval)
        text = " ".join(words).strip()
        text = text.replace("−", "-")
        avg_conf = (sum(confs) / len(confs)) if confs else 0.0
        return text, avg_conf

    def _extract_pdf_text_in_rect(self, hs: Hotspot) -> Optional[str]:
        if not (self.bp and self.bp.is_pdf and fitz is not None and getattr(self.bp, "_doc", None) is not None):
            return None
        try:
            page = self.bp._doc.load_page(int(hs.page))
            scale_display = float(getattr(self.bp, "scale", 2.0) or 2.0)
            rect_pts = fitz.Rect(
                float(hs.x) / scale_display,
                float(hs.y) / scale_display,
                (float(hs.x) + float(max(1.0, hs.w))) / scale_display,
                (float(hs.y) + float(max(1.0, hs.h))) / scale_display,
            )
            txt = page.get_text("text", clip=rect_pts) or ""
            txt = txt.strip().replace("−", "-")
            return txt if txt else None
        except Exception:
            return None

    def _ocr_selected_row(self):
        if pytesseract is None:
            QMessageBox.warning(self, "OCR", "pytesseract is not installed. Install pytesseract to use OCR.")
            return
        if not self.hotspots or self.table.rowCount() == 0:
            QMessageBox.information(self, "OCR", "No hotspots available.")
            return
        indexes = self.table.selectionModel().selectedRows()
        if not indexes:
            QMessageBox.information(self, "OCR", "Select a hotspot row first.")
            return
        row = indexes[0].row()
        if row < 0 or row >= len(self.hotspots):
            return
        hs = self.hotspots[row]

        native_txt = self._extract_pdf_text_in_rect(hs)
        if native_txt:
            text = native_txt.splitlines()[0].strip()
            text = text.replace("−", "-")
            it = self.table.item(row, 3)
            if it is None:
                it = QTableWidgetItem("")
                self.table.setItem(row, 3, it)
            it.setText(text)
            self.table.setCurrentCell(row, 3)
            self.status.showMessage("Extracted native PDF text (no OCR)")
            return

        bgr = self._render_hotspot_clip(hs, dpi=300)
        if bgr is None or bgr.size == 0:
            QMessageBox.warning(self, "OCR", "Failed to capture hotspot image for OCR.")
            return
        prep = self._prep_for_ocr(bgr)
        try:
            text, conf = self._ocr_text(prep, digits_only=True)
        except Exception as e:
            QMessageBox.critical(self, "OCR", f"OCR failed: {e}")
            return
        it = self.table.item(row, 3)
        if it is None:
            it = QTableWidgetItem("")
            self.table.setItem(row, 3, it)
        it.setText(text)
        self.table.setCurrentCell(row, 3)
        shown = text if len(text) <= 40 else (text[:37] + "…")
        suffix = " (no OpenCV)" if cv2 is None else ""
        self.status.showMessage(f"OCR: '{shown}' (conf ≈ {int(round(conf))}){suffix}")

    def _delete_selected_rows(self):
        if not self.hotspots or self.table.rowCount() == 0:
            QMessageBox.information(self, "Delete", "No hotspots to delete.")
            return
        sel = self.table.selectionModel().selectedRows()
        rows = sorted((idx.row() for idx in sel), reverse=True)
        rows = [r for r in rows if 0 <= r < len(self.hotspots)]
        if not rows:
            QMessageBox.information(self, "Delete", "Select one or more rows to delete.")
            return
        resp = QMessageBox.question(
            self,
            "Confirm delete",
            f"Delete {len(rows)} selected row(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return
        for r in rows:
            self.hotspots.pop(r)
        if self.mode == "ballooning":
            self._save_hotspots_csv()
        self._refresh_table()
        if self._balloons_on and self.bp:
            self._refresh_balloons_for_page()
        new_row = min(rows[-1] if rows else 0, self.table.rowCount() - 1)
        if new_row >= 0:
            self.table.selectRow(new_row)
            if new_row < len(self.hotspots):
                hs = self.hotspots[new_row]
                if self.bp and self.bp.is_pdf and hs.page != self.current_page:
                    self._show_page(hs.page)
                rect = QRectF(hs.x, hs.y, max(2, hs.w), max(2, hs.h))
                self.view.center_on_rect(rect, hs.zoom)
        self.status.showMessage("Deleted row(s)")

    def _toggle_pick(self, on: bool):
        if self.mode != "ballooning":
            QMessageBox.information(self, "Pick mode", "Pick is available only in Ballooning mode.")
            try:
                self.act_pick.setChecked(False)
            except Exception:
                pass
            return
        if not self.bp:
            QMessageBox.information(self, "Pick mode", "Load a blueprint first.")
            try:
                self.act_pick.setChecked(False)
            except Exception:
                pass
            return
        self.view.set_pick_mode(on)
        self.status.showMessage("Pick mode: click-drag a rectangle on the print" if on else "Pick mode off")

    def _picked_rect(self, rect: QRectF):
        hs_id = self._generate_hotspot_id()
        zoom = 220
        hs = Hotspot(
            id=hs_id.strip(),
            page=self.current_page,
            x=float(rect.x()), y=float(rect.y()),
            w=float(rect.width()), h=float(rect.height()),
            zoom=float(zoom),
            method="", result="", nominal=None, lsl=None, usl=None
        )
        self.hotspots.append(hs)
        if self.mode == "ballooning":
            self._save_hotspots_csv()
        self._refresh_table()
        if self._balloons_on and hs.page == self.current_page:
            self._refresh_balloons_for_page()
            self.view.highlight_balloon(hs.id)
        self.status.showMessage(f"Added hotspot {hs_id} on page {self.current_page}")
        self.view.flash_rect(rect)

    def _generate_hotspot_id(self) -> str:
        prefix = "HS-"
        used = {hs.id for hs in self.hotspots}
        n = 1
        while True:
            cid = f"{prefix}{n:03d}"
            if cid not in used:
                return cid
            n += 1

    def _save_hotspots_csv(self):
        """Rewrite the global geometry/methods/limits CSV (no serial)."""
        path = self.hotspots_path_global
        if not path:
            return
        tmp = f"{path}.tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id","page","x","y","w","h","zoom","method","result","nominal","lsl","usl"])
            for hs in self.hotspots:
                def fmt(v):
                    return "" if v is None else (f"{v:g}")
                w.writerow([
                    hs.id, f"{hs.page}",
                    fmt(hs.x), fmt(hs.y),
                    fmt(hs.w), fmt(hs.h),
                    fmt(hs.zoom),
                    hs.method or "", hs.result or "",
                    fmt(hs.nominal),
                    fmt(hs.lsl),
                    fmt(hs.usl),
                ])
        os.replace(tmp, path)

    def _save_results_csv(self):
        """Write per-serial results overlay (id,result) only, merging with existing non-empty values to avoid accidental clears."""
        if not self.blueprint_path or not self.work_order:
            return
        path = self.results_path_wo or self._results_path_for_blueprint(self.blueprint_path, self.work_order)
        self.results_path_wo = path
        # Load existing results (if any)
        existing: dict[str, str] = {}
        if os.path.exists(path):
            try:
                with open(path, newline="", encoding="utf-8-sig") as f:
                    r = csv.DictReader(f)
                    for row in r:
                        rid = str(row.get("id", "")).strip()
                        res = str(row.get("result", "")).strip()
                        if rid:
                            existing[rid] = res
            except Exception:
                existing = {}
        # Merge: prefer in-memory non-empty; otherwise keep existing
        merged: dict[str, str] = {}
        hs_ids = [hs.id for hs in self.hotspots]
        for hs in self.hotspots:
            val = (hs.result or "").strip()
            merged[hs.id] = val if val != "" else existing.get(hs.id, "")
        # Preserve any extra IDs from the existing file (not in current hotspots)
        for rid, res in existing.items():
            if rid not in merged:
                merged[rid] = res
        # Write merged
        tmp = f"{path}.tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "result"])
            # Keep order: current hotspots first, then any extras from existing
            for rid in hs_ids:
                w.writerow([rid, merged.get(rid, "")])
            for rid, res in merged.items():
                if rid not in hs_ids:
                    w.writerow([rid, res])
        os.replace(tmp, path)

    def _export_csv(self):
        """Export currently visible (filtered) rows to a user-selected CSV file."""
        base_source = self.results_path_wo if (self.mode == "inspection" and self.results_path_wo) else (self.hotspots_path_global or "hotspots")
        base = os.path.splitext(base_source)[0]
        default_path = f"{base}.export.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export filtered hotspots", default_path, "CSV (*.csv)")
        if not path:
            return
        try:
            rows = self.table.rowCount()
            count = 0
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["id","page","x","y","w","h","zoom","method","result","nominal","lsl","usl"])
                def fmt(v):
                    return "" if v is None else (f"{v:g}")
                for r in range(rows):
                    if self.table.isRowHidden(r):
                        continue
                    if r < 0 or r >= len(self.hotspots):
                        continue
                    hs = self.hotspots[r]
                    w.writerow([
                        hs.id, f"{hs.page}",
                        fmt(hs.x), fmt(hs.y),
                        fmt(hs.w), fmt(hs.h),
                        fmt(hs.zoom),
                        hs.method or "", hs.result or "",
                        fmt(hs.nominal),
                        fmt(hs.lsl),
                        fmt(hs.usl),
                    ])
                    count += 1
            self.status.showMessage(f"Exported {count} row(s) → {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))

    def _row_selected(self, selected=None, deselected=None):
        if not self.bp or not self.hotspots:
            return
        indexes = self.table.selectionModel().selectedRows()
        if not indexes:
            return
        row = indexes[0].row()
        if row < 0 or row >= len(self.hotspots):
            return

        hs = self.hotspots[row]

        # switch page if needed (PDFs)
        if self.bp.is_pdf and hs.page != self.current_page:
            self._show_page(hs.page)

        rect = QRectF(hs.x, hs.y, max(2, hs.w), max(2, hs.h))
        self.view.center_on_rect(rect, hs.zoom)

        if self._balloons_on:
            self.view.highlight_balloon(hs.id)

        # focus Result column for quick entry (column 3 after adding Inspection Method)
        self.table.scrollTo(self.table.model().index(row, 0))
        self.table.setCurrentCell(row, 3)

    def closeEvent(self, event):
        # Persist splitter sizes so layout is restored on next launch
        try:
            if hasattr(self, "splitter"):
                self.settings.setValue("splitterSizes", self.splitter.sizes())
        except Exception:
            pass
        super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
