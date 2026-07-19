import sys
from PySide6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PySide6.QtCore import Qt


class TitanHUD(QWidget):
    def __init__(self):
        super().__init__()

        # ---- window behavior flags ----
        self.setWindowFlags(
            Qt.FramelessWindowHint |      # no title bar/border
            Qt.WindowStaysOnTopHint |     # always on top
            Qt.Tool                        # don't show in taskbar
        )
        self.setAttribute(Qt.WA_TranslucentBackground)  # transparent background

        # ---- layout ----
        layout = QVBoxLayout()
        label = QLabel("TITAN HUD")
        label.setStyleSheet("""
            color: cyan;
            font-size: 28px;
            font-weight: bold;
            background-color: rgba(0, 0, 0, 120);
            padding: 12px;
            border-radius: 10px;
        """)
        layout.addWidget(label)
        self.setLayout(layout)

        # ---- position + size ----
        self.setGeometry(50, 50, 250, 80)   # x, y, width, height


app = QApplication(sys.argv)
hud = TitanHUD()
hud.show()
sys.exit(app.exec())