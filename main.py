import sys, os, csv, shutil, glob
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF, pyqtSignal, QSettings, QSize, QMarginsF, QUrl
import sys, time, errno, traceback, re
from PyQt6.QtGui import QAction, QPixmap, QImage, QPainter, QCursor, QFont, QPen, QBrush, QColor, QKeySequence, QIcon, QShortcut, QTextDocument, QTextOption, QPageSize, QPageLayout, QDesktopServices
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QTableWidget, QTableWidgetItem, QToolBar,
    QVBoxLayout, QFileDialog, QMessageBox, QLabel, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsRectItem, QHBoxLayout, QPushButton, QInputDialog,
    QStatusBar, QGraphicsEllipseItem, QGraphicsSimpleTextItem, QStyledItemDelegate, QComboBox, QLineEdit, QDockWidget,
    QDialog, QFormLayout, QDoubleSpinBox, QDialogButtonBox, QSpinBox
)
from PyQt6.QtPrintSupport import QPrinter
import html

# ---- Matplotlib (optional) ----
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    _HAVE_MPL = True
except Exception:
    FigureCanvas = None
    Figure = None
    _HAVE_MPL = False

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

# ---- Optional global exception logging ----
def _install_exception_hook(log_name: str = "InspectionApp-error.log"):
    """Install a sys.excepthook that logs unhandled exceptions to a file
    and shows a friendly dialog instead of crashing.
    """
    def _hook(exc_type, exc, tb):
        try:
            import datetime
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Prefer current working directory for the log by default
            log_path = os.path.join(os.getcwd(), log_name)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] Unhandled exception\n")
                traceback.print_exception(exc_type, exc, tb, file=f)
        except Exception:
            pass
        try:
            QMessageBox.critical(None, "Unexpected error",
                                 "An unexpected error occurred. Details were logged to\n"
                                 f"{log_name}")
        except Exception:
            pass
    sys.excepthook = _hook

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
        # persistent selection rectangle (stays as long as a row is selected)
        self._selection: Optional[QGraphicsRectItem] = None
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

    def set_balloon_size(self, size: float, update_existing: bool = True):
        """Set default balloon diameter in scene coordinates and optionally update existing balloons."""
        try:
            size = float(size)
        except Exception:
            size = self._balloon_size
        self._balloon_size = max(12.0, min(200.0, size))
        if not update_existing:
            return
        # Update existing ellipse geometry and re-center label text
        for ellipse in self._balloons.values():
            try:
                ellipse.setRect(0, 0, self._balloon_size, self._balloon_size)
                # Adjust any child text label
                for child in getattr(ellipse, 'childItems', lambda: [])():
                    if isinstance(child, QGraphicsSimpleTextItem):
                        f = child.font()
                        # Scale font roughly with diameter; keep within reasonable bounds
                        pt = max(8.0, min(22.0, self._balloon_size * 0.33))
                        f.setPointSizeF(pt)
                        child.setFont(f)
                        tr = child.boundingRect()
                        child.setPos((self._balloon_size - tr.width()) / 2, (self._balloon_size - tr.height()) / 2 - 1)
            except Exception:
                pass

    def set_pixmap(self, pm: QPixmap):
        self.scene().clear()
        # Reset references to any cleared items to avoid double-removal warnings
        self._highlight = None
        self._selection = None
        self._rubber = None
        self._balloons.clear()
        self._pix_item = self.scene().addPixmap(pm)
        self.scene().setSceneRect(self._pix_item.boundingRect())
        self.resetTransform()
        self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)
        # Preserve current pick mode drag behavior after pixmap resets (page changes reload pixmap)
        self.setDragMode(QGraphicsView.DragMode.NoDrag if self._picking else QGraphicsView.DragMode.ScrollHandDrag)

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
        # If a persistent selection exists, update it instead of creating a transient flash
        if self._selection is not None and self._selection.scene() is not None:
            try:
                self._selection.setRect(rect)
                self._selection.setVisible(True)
            except Exception:
                pass
            return
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

    # ---- persistent selection API ----
    def set_selection_rect(self, rect: QRectF):
        # Create or update a persistent selection rectangle
        try:
            if self._selection is None or self._selection.scene() is None:
                self._selection = QGraphicsRectItem(rect)
                pen = QPen(Qt.GlobalColor.red, 2)
                self._selection.setPen(pen)
                # Transparent fill for selection; outline only
                self._selection.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                self._selection.setZValue(25)
                self.scene().addItem(self._selection)
            else:
                self._selection.setRect(rect)
                self._selection.setVisible(True)
        except Exception:
            pass

    def clear_selection_rect(self):
        try:
            if self._selection is not None and self._selection.scene() is not None:
                self.scene().removeItem(self._selection)
        except Exception:
            pass
        self._selection = None

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
            ellipse.setRect(0, 0, size, size)
            ellipse.setPen(QPen(Qt.GlobalColor.red, 3))
            ellipse.setBrush(QBrush(Qt.GlobalColor.white))
            ellipse.setZValue(10)
            ellipse.setPos(pos)
            ellipse.setToolTip(hs_id)
            # text centered inside ellipse
            text = QGraphicsSimpleTextItem(label, ellipse)
            f = QFont(); f.setBold(True)
            # Scale font with balloon size; base 34px -> ~11pt
            try:
                f.setPointSizeF(max(8.0, min(22.0, size * 0.33)))
            except Exception:
                f.setPointSizeF(11)
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
                try:
                    # Reflect cursor to avoid hand cursor in pick mode
                    item.setCursor(QCursor(Qt.CursorShape.OpenHandCursor) if enabled else QCursor(Qt.CursorShape.CrossCursor))
                except Exception:
                    pass

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

# ---------- Pop-out Preview Window ----------
class PreviewWindow(QMainWindow):
    """A lightweight window hosting an ImageView for a second-monitor preview."""
    def __init__(self, on_close=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Blueprint Preview")
        self.view = ImageView(self)
        self.setCentralWidget(self.view)
        self._on_close_cb = on_close
        # Restore geometry
        try:
            self._settings = QSettings("InspectionApp", "PreviewWindow")
            geom = self._settings.value("geometry")
            if geom:
                # PyQt will accept the stored QByteArray directly
                self.restoreGeometry(geom)
        except Exception:
            pass

    def closeEvent(self, event):
        # Save geometry and notify main window
        try:
            if hasattr(self, "_settings"):
                self._settings.setValue("geometry", self.saveGeometry())
        except Exception:
            pass
        try:
            if callable(self._on_close_cb):
                self._on_close_cb()
        except Exception:
            pass
        super().closeEvent(event)

# ---------- History / SPC Dock ----------
class HistoryProvider:
    """Aggregates historical numeric results for a hotspot across all per-serial results CSVs for the current blueprint."""
    def __init__(self, blueprint_path_getter):
        # blueprint_path_getter: callable returning current blueprint path
        self._get_bp = blueprint_path_getter

    def _result_files(self) -> list[str]:
        bp = self._get_bp() or ""
        if not bp:
            return []
        # Pattern: blueprint.ext.*.results.csv
        pattern = f"{bp}.*.results.csv"
        try:
            files = sorted(glob.glob(pattern))
        except Exception:
            files = []
        return files

    def load_values(self, hotspot_id: str, parse_func) -> list[float]:
        vals: list[float] = []
        if not hotspot_id:
            return vals
        for fp in self._result_files():
            try:
                with open(fp, newline="", encoding="utf-8-sig") as f:
                    r = csv.DictReader(f)
                    for row in r:
                        rid = str(row.get("id", "")).strip()
                        if rid != hotspot_id:
                            continue
                        res = str(row.get("result", "")).strip()
                        v = parse_func(res)
                        if v is not None:
                            vals.append(v)
            except Exception:
                continue
        return vals

    def _work_order_from_path(self, results_path: str) -> str:
        """Extract the work order token from a results CSV path built like
        '<blueprint>.<workorder>.results.csv'. Falls back gracefully to basename parsing.
        """
        try:
            bp = self._get_bp() or ""
            if bp and results_path.startswith(bp + ".") and results_path.endswith(".results.csv"):
                return results_path[len(bp) + 1 : -len(".results.csv")]
            # Fallback using basenames
            import os as _os
            base_bp = _os.path.basename(bp)
            base_fp = _os.path.basename(results_path)
            prefix = base_bp + "."
            suffix = ".results.csv"
            if base_fp.startswith(prefix) and base_fp.endswith(suffix):
                return base_fp[len(prefix) : -len(suffix)]
        except Exception:
            pass
        return ""

    def load_series(self, hotspot_id: str, parse_func) -> list[tuple[float, str]]:
        """Return list of (value, work_order) pairs for this hotspot across all result files."""
        out: list[tuple[float, str]] = []
        if not hotspot_id:
            return out
        for fp in self._result_files():
            wo = self._work_order_from_path(fp)
            try:
                with open(fp, newline="", encoding="utf-8-sig") as f:
                    r = csv.DictReader(f)
                    for row in r:
                        rid = str(row.get("id", "")).strip()
                        if rid != hotspot_id:
                            continue
                        res = str(row.get("result", "")).strip()
                        v = parse_func(res)
                        if v is not None:
                            out.append((v, wo))
            except Exception:
                continue
        return out

class HistoryDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("History / SPC", parent)
        self.setObjectName("HistorySPCDock")  # for QSettings/restore
        self._mw: MainWindow | None = None
        self.provider: HistoryProvider | None = None
        self._current_hotspot: Optional[Hotspot] = None
        self._hover_bind_id: Optional[int] = None
        self._hover_ctx = {
            'artist': None,
            'annot': None,
            'ax': None,
            'series': [],  # list of (x, y, work_order)
        }

        self.main_widget = QWidget()
        self.vbox = QVBoxLayout(self.main_widget)
        self.lbl_title = QLabel("Select a hotspot to view history")
        self.lbl_title.setWordWrap(True)
        self.vbox.addWidget(self.lbl_title)
        self.stats_label = QLabel("")
        self.stats_label.setWordWrap(True)
        self.vbox.addWidget(self.stats_label)

        if _HAVE_MPL:
            self.fig = Figure(figsize=(5, 4), constrained_layout=True)
            self.canvas = FigureCanvas(self.fig)
            self.vbox.addWidget(self.canvas, 1)
            # Connect hover handler
            try:
                self._hover_bind_id = self.canvas.mpl_connect('motion_notify_event', self._on_motion)
            except Exception:
                self._hover_bind_id = None
        else:
            self.fig = None
            self.canvas = None
            warn = QLabel("matplotlib not installed.\nRun: pip install matplotlib\nThe XmR chart will appear after installation.")
            warn.setStyleSheet("color: #b00; font-weight: bold;")
            self.vbox.addWidget(warn)

        self.setWidget(self.main_widget)
        self.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea)

    def attach(self, main_window: 'MainWindow'):
        self._mw = main_window
        self.provider = HistoryProvider(lambda: main_window.blueprint_path)

    def update_for_hotspot(self, hs: Hotspot, mw: 'MainWindow'):
        self._current_hotspot = hs
        if self.provider is None:
            return
        parse = mw._parse_value if hasattr(mw, '_parse_value') else (lambda s: None)
        series = self.provider.load_series(hs.id, parse)
        values = [v for (v, _wo) in series]
        self.lbl_title.setText(f"Hotspot <b>{hs.id}</b> — {len(values)} historical reading(s)")
        self._render_spc(series, hs)

    # ---- SPC rendering ----
    def _render_spc(self, series: list[tuple[float, str]], hs: Hotspot):
        if not _HAVE_MPL or self.fig is None:
            # only update stats label
            self.stats_label.setText(self._stats_text([v for (v, _wo) in series], hs))
            return
        ax_ind = self.fig.get_axes()[0] if self.fig.get_axes() else self.fig.add_subplot(211)
        ax_mr = self.fig.get_axes()[1] if len(self.fig.get_axes()) > 1 else self.fig.add_subplot(212)
        ax_ind.clear(); ax_mr.clear()
        values = [v for (v, _wo) in series]
        if not values:
            ax_ind.text(0.5, 0.5, "No numeric results yet", ha='center', va='center')
            ax_mr.text(0.5, 0.5, "", ha='center', va='center')
            self.stats_label.setText("No numeric results available.")
            self.canvas.draw(); return
        import math
        import statistics as stats
        n = len(values)
        mr = [abs(values[i] - values[i-1]) for i in range(1, n)] if n > 1 else []
        mean_x = stats.fmean(values)
        avg_mr = stats.fmean(mr) if mr else 0.0
        sigma = (avg_mr / 1.128) if avg_mr > 0 else 0.0  # Wheeler constant for n=2
        ucl_x = mean_x + 3 * sigma
        lcl_x = mean_x - 3 * sigma
        # Individuals chart
        x = list(range(1, n+1))
        ax_ind.plot(x, values, marker='o')
        # Transparent scatter overlay for hover hit-testing
        sc = ax_ind.scatter(x, values, s=40, alpha=0.0)
        # Store hover context
        self._hover_ctx['artist'] = sc
        self._hover_ctx['ax'] = ax_ind
        self._hover_ctx['series'] = [(xi, yi, series[i][1]) for i, (xi, yi) in enumerate(zip(x, values))]
        # Create/update annotation
        try:
            if self._hover_ctx.get('annot') is None:
                annot = ax_ind.annotate("", xy=(0,0), xytext=(12,12), textcoords="offset points",
                                        bbox=dict(boxstyle="round", fc="w", ec="#333", alpha=0.9))
                annot.set_visible(False)
                self._hover_ctx['annot'] = annot
        except Exception:
            pass
        ax_ind.axhline(mean_x, color='green', linestyle='--', label='Mean')
        if sigma > 0:
            ax_ind.axhline(ucl_x, color='red', linestyle=':')
            ax_ind.axhline(lcl_x, color='red', linestyle=':')
        if hs.lsl is not None:
            ax_ind.axhline(hs.lsl, color='#b060ff', linestyle='-.', label='LSL')
        if hs.usl is not None:
            ax_ind.axhline(hs.usl, color='#b060ff', linestyle='-.', label='USL')
        ax_ind.set_title("XmR Individuals")
        ax_ind.set_xlabel("Observation")
        ax_ind.set_ylabel("Value")
        ax_ind.legend(loc='best', fontsize='small')
        # Moving Range chart
        if mr:
            ax_mr.plot(range(2, n+1), mr, marker='s')
            ax_mr.axhline(avg_mr, color='green', linestyle='--', label='Mean MR')
            ucl_mr = avg_mr * 3.267  # Wheeler constant
            ax_mr.axhline(ucl_mr, color='red', linestyle=':')
            ax_mr.set_title("Moving Range")
            ax_mr.set_xlabel("Observation")
            ax_mr.set_ylabel("Range")
            ax_mr.legend(loc='best', fontsize='small')
        else:
            ax_mr.text(0.5, 0.5, "Need >=2 values for MR", ha='center', va='center')
            ax_mr.set_axis_off()
        self.stats_label.setText(self._stats_text(values, hs, mean_x, sigma))
        self.canvas.draw()

    def _stats_text(self, values: list[float], hs: Hotspot, mean_x: Optional[float] = None, sigma: Optional[float] = None) -> str:
        if not values:
            return ""
        import statistics as stats
        n = len(values)
        mean_x = mean_x if mean_x is not None else stats.fmean(values)
        stdev = stats.pstdev(values) if n > 1 else 0.0
        lsl, usl = hs.lsl, hs.usl
        cp = cpk = None
        if lsl is not None and usl is not None and sigma is not None and sigma > 0:
            cp = (usl - lsl) / (6 * sigma)
            cpu = (usl - mean_x) / (3 * sigma)
            cpl = (mean_x - lsl) / (3 * sigma)
            cpk = min(cpu, cpl)
        parts = [f"n={n}", f"mean={mean_x:g}"]
        parts.append(f"σ_ind≈{sigma:g}" if sigma is not None else f"stdev={stdev:g}")
        if cp is not None:
            parts.append(f"Cp={cp:.2f}")
        if cpk is not None:
            parts.append(f"Cpk={cpk:.2f}")
        return " · ".join(parts)

    # ---- hover handler ----
    def _on_motion(self, event):
        try:
            ctx = self._hover_ctx
            artist = ctx.get('artist')
            annot = ctx.get('annot')
            ax = ctx.get('ax')
            series = ctx.get('series') or []
            if artist is None or annot is None or ax is None:
                return
            if event.inaxes != ax:
                if annot.get_visible():
                    annot.set_visible(False)
                    self.canvas.draw_idle()
                return
            contains, info = artist.contains(event)
            if not contains:
                if annot.get_visible():
                    annot.set_visible(False)
                    self.canvas.draw_idle()
                return
            ind = info.get('ind')
            if not ind:
                return
            i = ind[0]
            if i < 0 or i >= len(series):
                return
            xi, yi, wo = series[i]
            annot.xy = (xi, yi)
            label = f"WO: {wo if wo else '(unknown)'}\nValue: {yi:g}\nObs: {xi}"
            annot.set_text(label)
            annot.set_visible(True)
            self.canvas.draw_idle()
        except Exception:
            pass

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
        self.preview_win: Optional[PreviewWindow] = None
        self.history_dock: Optional[HistoryDock] = None

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
        # Pop-out preview action
        self.act_preview = QAction(QIcon.fromTheme("window-new"), "Pop out Preview", self)
        self.act_preview.setToolTip("Open a detachable preview window for a second monitor")
        self.act_preview.setCheckable(True)
        try:
            self.act_preview.setShortcut(QKeySequence("Ctrl+Shift+P"))
        except Exception:
            pass
        self.act_preview.setEnabled(False)
        tools.addAction(self.act_preview)
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
        # Top-right logo (optional): looks for axis_logo.(png|jpg) in app folder or assets/
        try:
            self.logo_label = QLabel()
            self.logo_label.setObjectName("AppLogoLabel")
            self.logo_label.setStyleSheet("#AppLogoLabel { padding-right: 6px; }")
            app_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(app_dir, "axis_logo.png"),
                os.path.join(app_dir, "axis_logo.jpg"),
                os.path.join(app_dir, "assets", "axis_logo.png"),
                os.path.join(app_dir, "assets", "axis_logo.jpg"),
            ]
            pm = QPixmap()
            for fp in candidates:
                if os.path.exists(fp):
                    pm = QPixmap(fp)
                    if not pm.isNull():
                        break
            if not pm.isNull():
                pm = pm.scaledToHeight(24, Qt.TransformationMode.SmoothTransformation)
                self.logo_label.setPixmap(pm)
            else:
                # Fallback text logo
                self.logo_label.setText("Axis")
                f = QFont(); f.setBold(True); f.setPointSizeF(12)
                self.logo_label.setFont(f)
            btns.addWidget(self.logo_label, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        except Exception:
            pass
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
        # Apply persisted balloon size if present
        try:
            bs = self.settings.value("Balloon/size", 34.0)
            bs = float(bs) if bs is not None else 34.0
            if hasattr(self.view, 'set_balloon_size'):
                self.view.set_balloon_size(bs, update_existing=False)
        except Exception:
            pass

        # menu
        file_menu = self.menuBar().addMenu("&File")
        act_open_bp = QAction("Open Blueprint…", self); act_open_bp.triggered.connect(self._choose_blueprint)
        act_open_csv = QAction("Open Hotspots/Results…", self); act_open_csv.triggered.connect(self._choose_hotspots)
        act_export = QAction("Export CSV…", self); act_export.triggered.connect(self._export_csv)
        act_export_pdf = QAction("Export Table to PDF…", self); act_export_pdf.triggered.connect(self._export_pdf)
        file_menu.addAction(act_open_bp); file_menu.addAction(act_open_csv); file_menu.addAction(act_export); file_menu.addAction(act_export_pdf)

        # Settings menu for default tolerances
        settings_menu = self.menuBar().addMenu("&Settings")
        act_def_tols = QAction("Default Tolerances…", self)
        act_def_tols.triggered.connect(self._edit_default_tolerances)
        settings_menu.addAction(act_def_tols)
        # Balloon size settings
        act_balloon_size = QAction("Balloon Size…", self)
        act_balloon_size.triggered.connect(self._edit_balloon_settings)
        settings_menu.addAction(act_balloon_size)

        # signals
        self.table.selectionModel().selectionChanged.connect(self._row_selected)
        self.act_fit.triggered.connect(lambda: self.view.fitInView(self.view.scene().itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio))
        self.act_pick.toggled.connect(self._toggle_pick)
        self.btn_balloons.toggled.connect(self._toggle_balloons)
        self.btn_ocr.clicked.connect(self._ocr_selected_row)
        self.act_preview.toggled.connect(self._toggle_preview_window)
        self._balloons_on = True

        # Table delete action (keyboard shortcut + context menu)
        self.act_delete = QAction("Delete Row", self)
        self.act_delete.setShortcut(QKeySequence.StandardKey.Delete)
        self.act_delete.triggered.connect(self._delete_selected_rows)
        self.table.addAction(self.act_delete)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)

        # start with blank preview and empty checklist
        self.status.showMessage("Ready. Open a blueprint and hotspots from the File menu to begin.")
        # Revert any advanced styling the user asked to undo (keep readability only)
        self._revert_table_styling()
        # Create history dock (initially hidden; user can toggle from View menu)
        self._setup_history_dock_action()
        self._maybe_create_history_dock()

        # ---- keyboard shortcuts for Pick (P) and Drag (D) ----
        try:
            self.shortcut_pick = QShortcut(QKeySequence("P"), self)
            self.shortcut_pick.setContext(Qt.ShortcutContext.ApplicationShortcut)
            self.shortcut_pick.activated.connect(lambda: self.act_pick.setChecked(True))
        except Exception:
            pass
        try:
            self.shortcut_drag = QShortcut(QKeySequence("D"), self)
            self.shortcut_drag.setContext(Qt.ShortcutContext.ApplicationShortcut)
            self.shortcut_drag.activated.connect(lambda: self.act_pick.setChecked(False))
        except Exception:
            pass

    # ---- robust file I/O helpers ----
    def _safe_atomic_csv_write(self, path: str, headers: list[str], rows_iterable, attempts: int = 6, sleep_s: float = 0.15):
        """Atomically write CSV to 'path' via a temp file in the same directory.
        Retries os.replace on Windows if the file is locked. Cleans up .tmp files.
        Raises on final failure so caller can show a user-facing error.
        """
        if not path:
            raise ValueError("Target path is empty")
        dir_path = os.path.dirname(os.path.abspath(path))
        tmp_path = f"{path}.tmp"

        if dir_path and not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        # Write temp file in same folder
        try:
            with open(tmp_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if headers:
                    w.writerow(headers)
                for row in rows_iterable:
                    w.writerow(row)
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise

        # Replace with retries for Windows locks
        last_err: Optional[BaseException] = None
        for _ in range(max(1, int(attempts))):
            try:
                os.replace(tmp_path, path)
                last_err = None
                break
            except (PermissionError, OSError) as e:
                err_no = getattr(e, "errno", None)
                if isinstance(e, PermissionError) or err_no in (errno.EACCES, errno.EPERM):
                    last_err = e
                    time.sleep(max(0.0, float(sleep_s)))
                    continue
                else:
                    last_err = e
                    break
            except Exception as e:
                last_err = e
                break

        # Cleanup leftover temp file, if any
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

        if last_err is not None:
            raise last_err

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
        # Toggle balloon drag capability (respect Pick mode)
        if hasattr(self, "view") and hasattr(self.view, "_balloons"):
            try:
                pick_on = bool(self.act_pick.isChecked()) if hasattr(self, "act_pick") else False
                self.view.set_balloons_movable(self.mode == "ballooning" and not pick_on)
            except Exception:
                pass
        # Preview window balloons follow same rule (but preview may be closed)
        if getattr(self, "preview_win", None) is not None:
            try:
                pick_on = bool(self.act_pick.isChecked()) if hasattr(self, "act_pick") else False
                self.preview_win.view.set_balloons_movable(self.mode == "ballooning" and not pick_on)
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
        self.table.setStyleSheet("")

    def _revert_table_styling(self):
        """Undo previously applied modern styling, returning to a simpler native look."""
        try:
            self.table.setStyleSheet("")
            self.table.setAlternatingRowColors(False)
            # Reset header stretch last section off (we can still allow manual resize)
            hh = self.table.horizontalHeader()
            hh.setStretchLastSection(False)
        except Exception:
            pass

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
            # Enable pop-out preview action when a blueprint is loaded
            if hasattr(self, "act_preview"):
                self.act_preview.setEnabled(True)
            # Sync preview if it's already open
            self._sync_preview_full()
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
            # Update history dock now that blueprint context changed
            self._refresh_history_dock()
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
            self._refresh_history_dock()
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
        # Keep preview in sync when page changes
        self._sync_preview_full()

    def _refresh_balloons_for_page(self):
        page_hs = [hs for hs in self.hotspots if hs.page == self.current_page]
        items: list[tuple[str, QRectF, str]] = []
        for idx, hs in enumerate(page_hs, start=1):
            rect = QRectF(hs.x, hs.y, max(2, hs.w), max(2, hs.h))
            items.append((hs.id, rect, str(idx)))
        self.view.set_balloons(items)
        self._update_status_counts()
        # Update balloons in preview window too
        if self.preview_win and self.preview_win.isVisible():
            if self._balloons_on:
                try:
                    self.preview_win.view.set_balloons(items)
                    pick_on = bool(self.act_pick.isChecked()) if hasattr(self, 'act_pick') else False
                    self.preview_win.view.set_balloons_movable(self.mode == "ballooning" and not pick_on)
                except Exception:
                    pass
            else:
                try:
                    self.preview_win.view.clear_balloons()
                except Exception:
                    pass

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
        # Mirror visibility in preview
        if self.preview_win and self.preview_win.isVisible():
            if on:
                self._refresh_balloons_for_page()  # will push to preview as well
            else:
                try:
                    self.preview_win.view.clear_balloons()
                except Exception:
                    pass

    def _collect_methods(self) -> list[str]:
        # Merge persisted history with current table values
        methods: list[str] = []
        seen: set[str] = set()
        # Load history from settings
        try:
            settings = getattr(self, 'settings', None) or QSettings("InspectionApp", "MainWindow")
            hist = settings.value("Methods/history", [], list)
            if isinstance(hist, list):
                for v in hist:
                    s = (str(v).strip())
                    if s and s not in seen:
                        methods.append(s); seen.add(s)
        except Exception:
            pass
        rows = self.table.rowCount()
        for r in range(rows):
            it = self.table.item(r, 2)
            val = (it.text().strip() if it else "")
            if val and val not in seen:
                methods.append(val)
                seen.add(val)
        return methods

    def _save_method_to_history(self, method: str):
        method = (method or "").strip()
        if not method:
            return
        try:
            settings = getattr(self, 'settings', None) or QSettings("InspectionApp", "MainWindow")
            hist = settings.value("Methods/history", [], list)
            if not isinstance(hist, list):
                hist = []
            # Put most recent first; cap list length
            new_hist = [method] + [v for v in hist if str(v).strip() and str(v).strip() != method]
            MAX = 20
            if len(new_hist) > MAX:
                new_hist = new_hist[:MAX]
            settings.setValue("Methods/history", new_hist)
        except Exception:
            pass

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
    def _count_decimals(self, s: str) -> int:
        if s is None:
            return 0
        t = str(s).strip()
        # Normalize unicode minus
        t = t.replace("−", "-")
        # Extract the first number-like token
        m = re.search(r"[-+]?((?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][-+]?\d+)?", t)
        if not m:
            return 0
        token = m.group(1)
        if "." not in token:
            return 0
        # Count digits to the right of the decimal point, preserving trailing zeros
        # Examples: '1.' -> 0, '1.0' -> 1, '1.000' -> 3
        decs = token.split(".", 1)[1]
        return len(decs)

    def _get_default_tol(self, decimals: int) -> Optional[float]:
        # clamp >=3 -> 3; allow 0 via dec0 setting
        d = decimals
        if d >= 3:
            d = 3
        # Use QSettings (MainWindow self.settings already exists)
        try:
            settings = getattr(self, 'settings', None) or QSettings("InspectionApp", "MainWindow")
            key_map = {0: "Tolerance/dec0", 1: "Tolerance/dec1", 2: "Tolerance/dec2", 3: "Tolerance/dec3"}
            if d not in key_map:
                return None
            key = key_map[d]
            val = settings.value(key, None)
            if val is None or str(val).strip() == "":
                defaults = {0: 0.0, 1: 0.03, 2: 0.01, 3: 0.005}
                return float(defaults.get(d, 0.0))
            return float(val)
        except Exception:
            defaults = {0: 0.0, 1: 0.03, 2: 0.01, 3: 0.005}
            return float(defaults.get(d, 0.0))
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
        """Return (nominal, plus_tol, minus_tol) from strings like:
           1.005+.005-.000   or   1.005 - .000 + .005
        Uses fullmatch, normalizes signs/Ø, and ignores trailing TYP/REF/units.
        """
        import re
        if not text:
            return None

        s = str(text).strip()
        # Normalize common symbols and strip diameter symbol
        s = (s.replace("Ø", "")
             .replace("＋", "+").replace("－", "-").replace("−", "-"))

        # Remove trailing annotations/units like TYP/REF/in/mm after the spec
        s = re.sub(r"\b(?:TYP|REF)\b.*$", "", s, flags=re.I).strip()
        s = re.sub(r"\s*(?:in|mm)\s*$", "", s, flags=re.I).strip()

        # float patterns
        fp  = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
        num = rf"(?P<nom>[-+]?{fp})"

        # Allow spaces between sign and digits: + .005 / - .000
        plus  = rf"\+\s*(?P<plus>{fp})"
        minus = rf"-\s*(?P<minus>{fp})"

        # nom +a -b
        m = re.fullmatch(rf"{num}\s*{plus}\s*{minus}", s)
        if m:
            return (
                float(m.group("nom")),
                float(m.group("plus")),
                float(m.group("minus")),
            )

        # nom -a +b
        m = re.fullmatch(rf"{num}\s*{minus}\s*{plus}", s)
        if m:
            return (
                float(m.group("nom")),
                float(m.group("plus")),
                float(m.group("minus")),
            )

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
            # Preserve previous non-empty method if the edit produced an empty string inadvertently
            prev = self.hotspots[row].method or ""
            if text == "" and prev:
                # Revert visual cell text and skip persistence/filter churn
                self._updating_table = True
                try:
                    item.setText(prev)
                finally:
                    self._updating_table = False
                return
            self.hotspots[row].method = text
            if self.mode == "ballooning":
                self._save_hotspots_csv()
            # Persist into history list
            if text:
                self._save_method_to_history(text)
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
                else:
                    # New branch: bare number auto-fill default tolerance (Ballooning mode only)
                    if self.mode == "ballooning" and hs.lsl is None and hs.usl is None and text != "":
                        # Ensure it is a bare number: parse value and ensure original text matches a simple number pattern
                        val = self._parse_value(text)
                        if val is not None:
                            # Reject if original had tolerance syntax (already handled above) or non-numeric noise
                            # Simple pattern: optional sign digits . digits optional; no ± or + tol parts
                            if re.fullmatch(r"[-+]?\d+(?:\.\d*)?|[-+]?\.\d+", text):
                                decs = self._count_decimals(text)
                                tol = self._get_default_tol(decs)
                                if tol is not None:
                                    if hs.nominal is None:
                                        hs.nominal = val
                                    if hs.lsl is None:
                                        hs.lsl = hs.nominal - tol
                                    if hs.usl is None:
                                        hs.usl = hs.nominal + tol
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

    # ---- Settings: Default Tolerances ----
    def _edit_default_tolerances(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Default Tolerances")
        layout = QVBoxLayout(dlg)
        form = QFormLayout()
        # Load current values or defaults
        v0 = self._get_default_tol(0)
        v1 = self._get_default_tol(1) or 0.03
        v2 = self._get_default_tol(2) or 0.01
        v3 = self._get_default_tol(3) or 0.005
        sp0 = QDoubleSpinBox(); sp0.setDecimals(6); sp0.setRange(0.0, 1e6); sp0.setSingleStep(0.001); sp0.setValue(float(v0 or 0.0))
        sp1 = QDoubleSpinBox(); sp1.setDecimals(6); sp1.setRange(0.0, 1e6); sp1.setSingleStep(0.001); sp1.setValue(float(v1))
        sp2 = QDoubleSpinBox(); sp2.setDecimals(6); sp2.setRange(0.0, 1e6); sp2.setSingleStep(0.001); sp2.setValue(float(v2))
        sp3 = QDoubleSpinBox(); sp3.setDecimals(6); sp3.setRange(0.0, 1e6); sp3.setSingleStep(0.001); sp3.setValue(float(v3))
        form.addRow("± for 0 decimals (e.g., 1)", sp0)
        form.addRow("± for 1 decimal (e.g., 1.0)", sp1)
        form.addRow("± for 2 decimals (e.g., 1.00)", sp2)
        form.addRow("± for 3 decimals (e.g., 1.000)", sp3)
        layout.addLayout(form)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            try:
                settings = getattr(self, 'settings', None) or QSettings("InspectionApp", "MainWindow")
                settings.setValue("Tolerance/dec0", sp0.value())
                settings.setValue("Tolerance/dec1", sp1.value())
                settings.setValue("Tolerance/dec2", sp2.value())
                settings.setValue("Tolerance/dec3", sp3.value())
                self.status.showMessage("Saved default tolerances")
            except Exception:
                pass

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

    # ---- Settings: Balloon Size ----
    def _edit_balloon_settings(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Balloon Size")
        layout = QVBoxLayout(dlg)
        form = QFormLayout()
        # Current size from view or default
        try:
            curr_size = float(getattr(self.view, '_balloon_size', 34.0))
        except Exception:
            curr_size = 34.0
        sp = QSpinBox(); sp.setRange(12, 200); sp.setSingleStep(2); sp.setValue(int(round(curr_size)))
        sp.setSuffix(" px")
        form.addRow("Balloon diameter", sp)
        layout.addLayout(form)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            size = float(sp.value())
            try:
                # Apply to main view
                if hasattr(self.view, 'set_balloon_size'):
                    self.view.set_balloon_size(size, update_existing=True)
                # Apply to preview if present
                if getattr(self, 'preview_win', None) is not None and hasattr(self.preview_win.view, 'set_balloon_size'):
                    self.preview_win.view.set_balloon_size(size, update_existing=True)
                # Persist setting
                self.settings.setValue("Balloon/size", size)
                self.status.showMessage(f"Balloon size set to {int(round(size))} px")
            except Exception:
                pass

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
        # While picking, prevent balloon movement to avoid accidental drags
        try:
            self.view.set_balloons_movable(False if on else (self.mode == "ballooning"))
        except Exception:
            pass
        # Preview balloons: movable only when not picking and in Ballooning mode
        if getattr(self, "preview_win", None) is not None:
            try:
                self.preview_win.view.set_balloons_movable(False if on else (self.mode == "ballooning"))
                # Mirror pick state into preview so picking works there too
                self.preview_win.view.set_pick_mode(on)
                # Wire preview picking to same handler (safe to connect multiple times guardedly)
                try:
                    # Avoid duplicate connections by disconnecting first if previously connected
                    self.preview_win.view.rectPicked.disconnect(self._picked_rect)
                except Exception:
                    pass
                self.preview_win.view.rectPicked.connect(self._picked_rect)
            except Exception:
                pass
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
        headers = ["id","page","x","y","w","h","zoom","method","result","nominal","lsl","usl"]
        def fmt(v):
            return "" if v is None else (f"{v:g}")
        def rows():
            for hs in self.hotspots:
                yield [
                    hs.id, f"{hs.page}",
                    fmt(hs.x), fmt(hs.y),
                    fmt(hs.w), fmt(hs.h),
                    fmt(hs.zoom),
                    hs.method or "", hs.result or "",
                    fmt(hs.nominal),
                    fmt(hs.lsl),
                    fmt(hs.usl),
                ]
        try:
            self._safe_atomic_csv_write(path, headers, rows())
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Save error", f"Could not save hotspots CSV:\n{path}\n\n{e}")

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
        headers = ["id", "result"]
        def rows():
            # Keep order: current hotspots first, then any extras from existing
            for rid in hs_ids:
                yield [rid, merged.get(rid, "")]
            for rid, res in merged.items():
                if rid not in hs_ids:
                    yield [rid, res]
        try:
            self._safe_atomic_csv_write(path, headers, rows())
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Save error", f"Could not save results CSV:\n{path}\n\n{e}")

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

    def _export_pdf(self):
        """Export full hotspots table (unfiltered) to a PDF with header metadata."""
        # Default filename based on the current CSV (results preferred, else hotspots), with .pdf extension
        csv_source = self.results_path_wo or self.hotspots_path_global or "hotspots.csv"
        try:
            base_pdf = os.path.splitext(csv_source)[0] + ".pdf"
        except Exception:
            base_pdf = "hotspots_table.pdf"
        path, _ = QFileDialog.getSaveFileName(self, "Export Table to PDF", base_pdf, "PDF (*.pdf)")
        if not path:
            return
        # Derive part number and work order from current file naming conventions.
        # part_number from blueprint file name (stem without extension)
        part_number = ""
        if self.blueprint_path:
            bp_base = os.path.basename(self.blueprint_path)
            part_number = os.path.splitext(bp_base)[0]
        # work_order from results CSV path if available; fall back to self.work_order
        work_order = self.work_order or ""
        if self.results_path_wo:
            inferred = self._infer_work_order_from_csv(self.results_path_wo, self.blueprint_path if self.blueprint_path else None)
            if inferred is not None:
                work_order = inferred or work_order
        try:
            doc = QTextDocument()
            doc.setDefaultFont(QFont("Sans", 9))
            header_html = "<h2 style='margin:4px 0'>Inspection Hotspots</h2>"
            meta = []
            if part_number:
                meta.append(f"<b>Part Number:</b> {html.escape(part_number)}")
            if work_order:
                meta.append(f"<b>Work Order:</b> {html.escape(work_order)}")
            if self.blueprint_path:
                meta.append(f"<b>Blueprint:</b> {html.escape(os.path.basename(self.blueprint_path))}")
            meta_html = ("<p style='margin:2px 0'>" + " &nbsp; · &nbsp; ".join(meta) + "</p>") if meta else ""
            ths = ["ID","Page","Inspection Method","Result","Nominal","LSL","USL","Status"]
            table_rows = []
            for hs in self.hotspots:
                status = self._compute_status(hs)
                row_cells = [
                    html.escape(hs.id or ""),
                    str(hs.page),
                    html.escape(hs.method or ""),
                    html.escape(hs.result or ""),
                    html.escape(self._fmt_num(hs.nominal)),
                    html.escape(self._fmt_num(hs.lsl)),
                    html.escape(self._fmt_num(hs.usl)),
                    html.escape(status)
                ]
                tds = "".join(f"<td style='padding:2px 6px;border:1px solid #aaa'>{c}</td>" for c in row_cells)
                # Row shading based on status
                if status == "PASS":
                    row_style = "background-color:#aaffaa;"
                elif status == "FAIL":
                    row_style = "background-color:#ffaaaa;"
                else:
                    row_style = "background-color:#ffeb96;"
                table_rows.append(f"<tr style='{row_style}'>" + tds + "</tr>")
            table_html = (
                "<table style='border-collapse:collapse;margin-top:6px'>"
                + "<thead><tr>"
                + "".join(f"<th style='background:#eee;padding:3px 6px;border:1px solid #aaa'>{html.escape(h)}</th>" for h in ths)
                + "</tr></thead><tbody>"
                + "".join(table_rows)
                + "</tbody></table>"
            )
            full_html = f"<html><body>{header_html}{meta_html}{table_html}</body></html>"
            doc.setHtml(full_html)
            printer = QPrinter(QPrinter.PrinterMode.HighResolution)
            printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
            printer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
            printer.setOutputFileName(path)
            layout = QPageLayout(printer.pageLayout())
            layout.setMargins(QMarginsF(15, 15, 15, 15))
            printer.setPageLayout(layout)
            doc.print(printer)
            self.status.showMessage(f"Exported PDF: {os.path.basename(path)}")
            # Confirmation dialog: open file?
            try:
                resp = QMessageBox.question(
                    self,
                    "Export Successful",
                    f"PDF was saved to:\n{path}\n\nWould you like to open it now?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if resp == QMessageBox.StandardButton.Yes:
                    # Try to open with system default viewer
                    opened = QDesktopServices.openUrl(QUrl.fromLocalFile(path))
                    if not opened:
                        QMessageBox.information(self, "Open PDF", "Couldn't open the PDF with the default viewer.")
            except Exception:
                pass
        except Exception as e:
            QMessageBox.critical(self, "Export PDF", f"Failed: {e}")

    def _row_selected(self, selected=None, deselected=None):
        # Commit any active cell editor first so edits persist before changing selection
        try:
            from PyQt6.QtWidgets import QAbstractItemDelegate
            ed = self.table.focusWidget()
            if ed is not None:
                # Only close editors that belong to this table to avoid framework warnings
                try:
                    p = ed
                    belongs = False
                    while p is not None:
                        if p is self.table or p is self.table.viewport():
                            belongs = True
                            break
                        p = p.parent()
                    if belongs:
                        try:
                            self.table.closeEditor(ed, QAbstractItemDelegate.EndEditHint.SubmitModelCache)
                        except Exception:
                            try:
                                ed.clearFocus()
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception:
            pass

        if not self.bp or not self.hotspots:
            return
        indexes = self.table.selectionModel().selectedRows()
        if not indexes:
            # No selection: clear selection rects in both views
            try:
                if hasattr(self.view, "clear_selection_rect"):
                    self.view.clear_selection_rect()
                if self.preview_win and self.preview_win.isVisible() and hasattr(self.preview_win.view, "clear_selection_rect"):
                    self.preview_win.view.clear_selection_rect()
            except Exception:
                pass
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
        # Persistent selection rectangle in main view
        try:
            if hasattr(self.view, "set_selection_rect"):
                self.view.set_selection_rect(rect)
        except Exception:
            pass

        if self._balloons_on:
            self.view.highlight_balloon(hs.id)

        # focus Result column for quick entry (column 3 after adding Inspection Method)
        self.table.scrollTo(self.table.model().index(row, 0))
        self.table.setCurrentCell(row, 3)
        # Sync selection/centering with preview
        if self.preview_win and self.preview_win.isVisible():
            try:
                # Do not reset the preview pixmap here (it clears balloons).
                # Assume the preview page was synced in _show_page/_sync_preview_full.
                self.preview_win.view.center_on_rect(rect, hs.zoom)
                if hasattr(self.preview_win.view, "set_selection_rect"):
                    self.preview_win.view.set_selection_rect(rect)
                if self._balloons_on:
                    self.preview_win.view.highlight_balloon(hs.id)
            except Exception:
                pass
        # Update History / SPC dock
        if getattr(self, 'history_dock', None) is not None and self.history_dock.isVisible():
            try:
                self.history_dock.update_for_hotspot(hs, self)
            except Exception:
                pass

    # ---- Preview sync & toggle ----
    def _toggle_preview_window(self, on: bool):
        if on:
            if not self.bp:
                QMessageBox.information(self, "Preview", "Load a blueprint first to open the preview.")
                try:
                    self.act_preview.setChecked(False)
                except Exception:
                    pass
                return
            if self.preview_win is None:
                self.preview_win = PreviewWindow(on_close=self._on_preview_closed, parent=None)
                try:
                    self.preview_win.view.set_context_key(self._balloon_context_key())
                    # Mirror current pick mode state into preview so picking works there too
                    pick_on = bool(self.act_pick.isChecked()) if hasattr(self, 'act_pick') else False
                    self.preview_win.view.set_pick_mode(pick_on)
                    # Set initial balloon movability in preview
                    self.preview_win.view.set_balloons_movable(self.mode == "ballooning" and not pick_on)
                    # Apply current balloon size to preview
                    try:
                        curr_size = float(getattr(self.view, '_balloon_size', 34.0))
                        if hasattr(self.preview_win.view, 'set_balloon_size'):
                            self.preview_win.view.set_balloon_size(curr_size, update_existing=False)
                    except Exception:
                        pass
                except Exception:
                    pass
            # Show preview without stealing focus and keep it behind the main window
            try:
                self.preview_win.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
            except Exception:
                pass
            self.preview_win.show()
            try:
                self.preview_win.lower()
            except Exception:
                pass
            self._sync_preview_full()
        else:
            if self.preview_win is not None:
                try:
                    self.preview_win.close()
                except Exception:
                    pass

    def _on_preview_closed(self):
        # Called when the pop-out window is closed (by user or programmatically)
        try:
            if hasattr(self, "act_preview") and self.act_preview.isChecked():
                self.act_preview.setChecked(False)
        except Exception:
            pass
        self.preview_win = None

    def _sync_preview_full(self):
        """Synchronize the preview window with the main view: page image, balloons, and selection."""
        if not (self.preview_win and self.preview_win.isVisible() and self.bp):
            return
        try:
            # Sync page image
            pm = self.bp.render_page(self.current_page)
            self.preview_win.view.set_pixmap(pm)
            self.preview_win.view.set_context_key(self._balloon_context_key())
            # Keep pick mode mirrored (page changes could reset drag cursor) and balloon movability
            if hasattr(self, 'act_pick'):
                pick_on = bool(self.act_pick.isChecked())
                self.preview_win.view.set_pick_mode(pick_on)
                self.preview_win.view.set_balloons_movable(self.mode == "ballooning" and not pick_on)
            # Sync balloons
            if self._balloons_on:
                page_hs = [hs for hs in self.hotspots if hs.page == self.current_page]
                items: list[tuple[str, QRectF, str]] = []
                for idx, hs in enumerate(page_hs, start=1):
                    rect = QRectF(hs.x, hs.y, max(2, hs.w), max(2, hs.h))
                    items.append((hs.id, rect, str(idx)))
                self.preview_win.view.set_balloons(items)
                pick_on = bool(self.act_pick.isChecked()) if hasattr(self, 'act_pick') else False
                self.preview_win.view.set_balloons_movable(self.mode == "ballooning" and not pick_on)
            else:
                self.preview_win.view.clear_balloons()
            # Sync selection highlighting and centering
            indexes = self.table.selectionModel().selectedRows()
            if indexes:
                row = indexes[0].row()
                if 0 <= row < len(self.hotspots):
                    hs = self.hotspots[row]
                    rect = QRectF(hs.x, hs.y, max(2, hs.w), max(2, hs.h))
                    self.preview_win.view.center_on_rect(rect, hs.zoom)
                    if hasattr(self.preview_win.view, "set_selection_rect"):
                        self.preview_win.view.set_selection_rect(rect)
                    if self._balloons_on:
                        self.preview_win.view.highlight_balloon(hs.id)
            else:
                # No selection: clear selection rect in preview
                if hasattr(self.preview_win.view, "clear_selection_rect"):
                    self.preview_win.view.clear_selection_rect()
        except Exception:
            pass

    # ---- History Dock wiring ----
    def _setup_history_dock_action(self):
        if getattr(self, 'act_history', None) is not None:
            return
        self.act_history = QAction("History / SPC", self)
        self.act_history.setCheckable(True)
        try:
            self.act_history.setShortcut(QKeySequence("Ctrl+Shift+H"))
        except Exception:
            pass
        self.act_history.toggled.connect(self._toggle_history_dock)
        # Ensure a View menu exists
        try:
            if not hasattr(self, 'view_menu') or self.view_menu is None:
                self.view_menu = self.menuBar().addMenu("&View")
            self.view_menu.addAction(self.act_history)
        except Exception:
            pass

    def _maybe_create_history_dock(self):
        if getattr(self, 'history_dock', None) is not None:
            return
        try:
            self.history_dock = HistoryDock(self)
            self.history_dock.attach(self)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.history_dock)
            self.history_dock.hide()
            # Sync action when visibility changes
            def _sync_action(vis: bool):
                try:
                    if getattr(self, 'act_history', None) is not None:
                        if self.act_history.isChecked() != bool(vis):
                            self.act_history.blockSignals(True)
                            self.act_history.setChecked(bool(vis))
                            self.act_history.blockSignals(False)
                except Exception:
                    pass
            self.history_dock.visibilityChanged.connect(_sync_action)
        except Exception:
            self.history_dock = None

    def _toggle_history_dock(self, on: bool):
        self._maybe_create_history_dock()
        if self.history_dock is None:
            return
        if on:
            self.history_dock.show()
            self._refresh_history_dock()
        else:
            self.history_dock.hide()

    def _refresh_history_dock(self):
        if self.history_dock is None or not self.history_dock.isVisible():
            return
        try:
            indexes = self.table.selectionModel().selectedRows()
            if indexes:
                row = indexes[0].row()
                if 0 <= row < len(self.hotspots):
                    hs = self.hotspots[row]
                    self.history_dock.update_for_hotspot(hs, self)
            else:
                self.history_dock.lbl_title.setText("Select a hotspot to view history")
                self.history_dock.stats_label.setText("")
        except Exception:
            pass

    def closeEvent(self, event):
        # Persist splitter sizes so layout is restored on next launch
        try:
            if hasattr(self, "splitter"):
                self.settings.setValue("splitterSizes", self.splitter.sizes())
            # Ensure preview window geometry is saved as well
            if self.preview_win is not None:
                try:
                    self.preview_win.close()
                except Exception:
                    pass
        except Exception:
            pass
        super().closeEvent(event)

def main():
    # Optionally install a global exception hook to log unexpected errors
    try:
        _install_exception_hook()
    except Exception:
        pass
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
