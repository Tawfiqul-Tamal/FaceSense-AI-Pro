import os
import time
import cv2
import numpy as np
import yaml
from pathlib import Path

from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSpacerItem,
    QSpinBox,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QGroupBox,
    QGridLayout,
    QMessageBox,
    QListWidget,
    QListWidgetItem,
)

from core.analysis_engine import AnalysisEngine
from core.camera_manager import CameraManager
from database.database import DatabaseManager
from analytics.report_generator import ReportGenerator
from analytics.charts import save_emotion_pie, save_age_histogram, save_head_pose_pie


class MainWindow(QMainWindow):
    def __init__(self, settings, config_path="config/settings.yaml"):
        super().__init__()
        self.settings = settings
        self.config_path = config_path
        self.setWindowTitle(self.settings["app"]["name"])
        self.setMinimumSize(self.settings["ui"]["width"], self.settings["ui"]["height"])
        self._apply_theme(self.settings["app"].get("theme", "dark"))

        self.camera_manager = CameraManager(self.settings["app"]["camera_id"])
        self.analysis_engine = AnalysisEngine(self.settings)
        self.database_manager = DatabaseManager(self.settings["database"]["path"])
        self.report_generator = ReportGenerator(self.settings)

        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_frame)
        self.camera_running = False
        self.last_summary = None
        self.last_saved_summary = None
        self.last_history_update = time.time()
        self.history_refresh_interval = self.settings["ui"].get("history_refresh_interval_sec", 3)

        self._build_ui()

    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(self._build_dashboard_tab(), "Dashboard")
        self.tab_widget.addTab(self._build_history_tab(), "History")
        self.tab_widget.addTab(self._build_analytics_tab(), "Analytics")
        self.tab_widget.addTab(self._build_reports_tab(), "Reports")
        self.tab_widget.addTab(self._build_settings_tab(), "Settings")

        main_layout.addWidget(self.tab_widget)

    def _build_dashboard_tab(self):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(16)

        title_label = QLabel(self.settings["app"]["name"])
        title_label.setObjectName("titleLabel")
        subtitle_label = QLabel(self.settings["app"]["subtitle"])
        subtitle_label.setObjectName("subtitleLabel")

        self.video_label = QLabel()
        self.video_label.setFixedSize(760, 520)
        self.video_label.setStyleSheet("border: 2px solid #2d3d65; border-radius: 12px;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Live Webcam Feed")

        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.toggle_camera)
        self.capture_button = QPushButton("Capture Image")
        self.capture_button.clicked.connect(self.capture_image)
        self.export_button = QPushButton("Export Report")
        self.export_button.clicked.connect(self.export_report)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.capture_button)
        button_layout.addWidget(self.export_button)

        left_layout.addWidget(title_label)
        left_layout.addWidget(subtitle_label)
        left_layout.addWidget(self.video_label)
        left_layout.addLayout(button_layout)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(16)

        status_group = QGroupBox("Current Session Summary")
        status_layout = QGridLayout(status_group)
        status_layout.setVerticalSpacing(18)
        status_layout.setHorizontalSpacing(12)

        self.emotion_value = QLabel("Unknown")
        self.age_value = QLabel("N/A")
        self.gender_value = QLabel("Unknown")
        self.attention_value = QLabel("N/A")
        self.stress_value = QLabel("N/A")
        self.head_pose_value = QLabel("Center")
        self.head_pose_icon = QLabel()
        self.head_pose_icon.setFixedSize(14, 14)
        self.head_pose_icon.setStyleSheet("border-radius: 7px; background: #42c26a;")
        self.head_pose_icon.setToolTip("Head pose status indicator")
        self.face_count_value = QLabel("0")

        head_pose_widget = QWidget()
        head_pose_layout = QHBoxLayout(head_pose_widget)
        head_pose_layout.setContentsMargins(0, 0, 0, 0)
        head_pose_layout.setSpacing(8)
        head_pose_layout.addWidget(self.head_pose_value)
        head_pose_layout.addWidget(self.head_pose_icon)

        rows = [
            ("Emotion:", self.emotion_value),
            ("Age:", self.age_value),
            ("Gender:", self.gender_value),
            ("Attention:", self.attention_value),
            ("Stress:", self.stress_value),
            ("Head Pose:", head_pose_widget),
            ("Faces Found:", self.face_count_value),
        ]

        for row_index, (label_text, value_widget) in enumerate(rows):
            status_layout.addWidget(QLabel(label_text), row_index, 0)
            value_widget.setObjectName("valueLabel")
            status_layout.addWidget(value_widget, row_index, 1)

        right_layout.addWidget(status_group)

        profile_group = QGroupBox("Operator Profile")
        profile_layout = QVBoxLayout(profile_group)
        self.profile_image = QLabel()
        self.profile_image.setFixedSize(160, 160)
        self.profile_image.setAlignment(Qt.AlignCenter)
        self.profile_image.setStyleSheet("border-radius: 80px; background: #1d2748;")
        self._load_profile_image()
        self.profile_name = QLabel(self.settings["app"]["author"])
        self.profile_name.setObjectName("valueLabel")
        self.profile_role = QLabel("AI Researcher | CSE Department")
        self.profile_role.setObjectName("subtitleLabel")
        profile_layout.addWidget(self.profile_image)
        profile_layout.addWidget(self.profile_name, 0, Qt.AlignCenter)
        profile_layout.addWidget(self.profile_role, 0, Qt.AlignCenter)
        right_layout.addWidget(profile_group)

        analytics_group = QGroupBox("Live Analytics")
        analytics_layout = QVBoxLayout(analytics_group)
        self.analytics_label = QLabel("Emotion and age charts export ready.")
        self.analytics_label.setWordWrap(True)
        self.history_label = QLabel("Latest session details will appear here.")
        self.history_label.setObjectName("valueLabel")
        self.history_label.setWordWrap(True)
        self.head_pose_hint = QLabel("Head pose indicator: green=center, yellow=left/right")
        self.head_pose_hint.setObjectName("subtitleLabel")
        self.head_pose_hint.setToolTip("Head pose status uses the icon color to indicate whether the user is centered or turned left/right.")
        analytics_layout.addWidget(self.analytics_label)
        analytics_layout.addWidget(self.history_label)
        analytics_layout.addWidget(self.head_pose_hint)
        right_layout.addWidget(analytics_group)

        layout.addWidget(left_panel, 3)
        layout.addWidget(right_panel, 2)

        return container

    def _build_history_tab(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)

        title_label = QLabel("History Dashboard")
        title_label.setObjectName("titleLabel")
        subtitle_label = QLabel("Review the latest captured sessions and analytics charts.")
        subtitle_label.setObjectName("subtitleLabel")

        self.history_list = QListWidget()
        self.history_list.setMinimumHeight(220)

        chart_group = QGroupBox("Analytics Charts")
        chart_layout = QVBoxLayout(chart_group)
        summary_label = QLabel("Emotion and age distribution charts are available after exporting session results.")
        summary_label.setObjectName("subtitleLabel")
        summary_label.setWordWrap(True)

        chart_row = QHBoxLayout()
        self.history_chart_pie = QLabel()
        self.history_chart_pie.setFixedSize(360, 300)
        self.history_chart_pie.setAlignment(Qt.AlignCenter)
        self.history_chart_pie.setStyleSheet("border: 1px solid #2d3d65; border-radius: 12px;")
        self.history_chart_pie.setText("Emotion chart preview")

        self.history_chart_age = QLabel()
        self.history_chart_age.setFixedSize(360, 300)
        self.history_chart_age.setAlignment(Qt.AlignCenter)
        self.history_chart_age.setStyleSheet("border: 1px solid #2d3d65; border-radius: 12px;")
        self.history_chart_age.setText("Age histogram preview")

        chart_row.addWidget(self.history_chart_pie)
        chart_row.addWidget(self.history_chart_age)

        button_row = QHBoxLayout()
        refresh_history_button = QPushButton("Refresh History")
        refresh_history_button.clicked.connect(self.load_history)
        refresh_charts_button = QPushButton("Refresh Charts")
        refresh_charts_button.clicked.connect(self.load_saved_charts)
        button_row.addWidget(refresh_history_button)
        button_row.addWidget(refresh_charts_button)

        chart_layout.addWidget(summary_label)
        chart_layout.addLayout(chart_row)
        chart_layout.addLayout(button_row)

        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addWidget(self.history_list)
        layout.addWidget(chart_group)

        return container

    def _build_analytics_tab(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)

        title_label = QLabel("Live Analytics")
        title_label.setObjectName("titleLabel")
        subtitle_label = QLabel("Visualize trends, session statistics, and distribution charts.")
        subtitle_label.setObjectName("subtitleLabel")

        self.analytics_summary_label = QLabel("Analytics summary will appear after the camera starts and data is recorded.")
        self.analytics_summary_label.setWordWrap(True)
        self.analytics_summary_label.setObjectName("valueLabel")

        button_row = QHBoxLayout()
        refresh_button = QPushButton("Refresh Analytics")
        refresh_button.clicked.connect(self.load_analytics)
        button_row.addWidget(refresh_button)
        button_row.addSpacerItem(QSpacerItem(24, 24, QSizePolicy.Expanding, QSizePolicy.Minimum))

        chart_group = QGroupBox("Trend Charts")
        chart_layout = QHBoxLayout(chart_group)
        self.analytics_chart_pie = QLabel()
        self.analytics_chart_pie.setFixedSize(360, 300)
        self.analytics_chart_pie.setAlignment(Qt.AlignCenter)
        self.analytics_chart_pie.setStyleSheet("border: 1px solid #2d3d65; border-radius: 12px;")
        self.analytics_chart_pie.setText("Emotion distribution")

        self.analytics_chart_age = QLabel()
        self.analytics_chart_age.setFixedSize(300, 260)
        self.analytics_chart_age.setAlignment(Qt.AlignCenter)
        self.analytics_chart_age.setStyleSheet("border: 1px solid #2d3d65; border-radius: 12px;")
        self.analytics_chart_age.setText("Age distribution")

        self.analytics_chart_pose = QLabel()
        self.analytics_chart_pose.setFixedSize(300, 260)
        self.analytics_chart_pose.setAlignment(Qt.AlignCenter)
        self.analytics_chart_pose.setStyleSheet("border: 1px solid #2d3d65; border-radius: 12px;")
        self.analytics_chart_pose.setText("Head pose distribution")

        chart_layout.addWidget(self.analytics_chart_pie)
        chart_layout.addWidget(self.analytics_chart_age)
        chart_layout.addWidget(self.analytics_chart_pose)

        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addWidget(self.analytics_summary_label)
        layout.addLayout(button_row)
        layout.addWidget(chart_group)

        return container

    def _build_reports_tab(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)

        title_label = QLabel("Report Center")
        title_label.setObjectName("titleLabel")
        subtitle_label = QLabel("Generate or open exported CSV, Excel, PDF, and image reports.")
        subtitle_label.setObjectName("subtitleLabel")

        self.reports_label = QLabel("Use Export Report button on the dashboard to create session analytics files.")
        self.reports_label.setWordWrap(True)

        button_layout = QHBoxLayout()
        self.open_export_button = QPushButton("Open Export Folder")
        self.open_export_button.clicked.connect(self.open_export_folder)
        self.refresh_exports_button = QPushButton("Refresh Report List")
        self.refresh_exports_button.clicked.connect(self.refresh_export_list)
        button_layout.addWidget(self.open_export_button)
        button_layout.addWidget(self.refresh_exports_button)

        self.export_files_label = QLabel("No exports available yet.")
        self.export_files_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addWidget(self.reports_label)
        layout.addLayout(button_layout)
        layout.addWidget(self.export_files_label)

        return container

    def _build_settings_tab(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)

        title_label = QLabel("Application Settings")
        title_label.setObjectName("titleLabel")
        subtitle_label = QLabel("Configure camera, export, and analytics preferences.")
        subtitle_label.setObjectName("subtitleLabel")

        form_layout = QFormLayout()
        form_layout.setLabelAlignment(Qt.AlignRight)

        self.camera_id_input = QSpinBox()
        self.camera_id_input.setRange(0, 10)
        self.camera_id_input.setValue(self.settings["app"].get("camera_id", 0))

        self.refresh_rate_input = QSpinBox()
        self.refresh_rate_input.setRange(10, 500)
        self.refresh_rate_input.setValue(self.settings["ui"].get("refresh_interval_ms", 30))

        self.export_dir_input = QLineEdit(os.path.join(os.getcwd(), self.settings["analytics"]["export_dir"]))
        self.export_dir_input.setReadOnly(True)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_export_folder)
        export_dir_widget = QWidget()
        export_dir_layout = QHBoxLayout(export_dir_widget)
        export_dir_layout.setContentsMargins(0, 0, 0, 0)
        export_dir_layout.addWidget(self.export_dir_input)
        export_dir_layout.addWidget(browse_button)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["dark", "light"])
        self.theme_combo.setCurrentText(self.settings["app"].get("theme", "dark"))
        self.theme_combo.currentTextChanged.connect(self.update_theme_preview)

        self.theme_preview_label = QLabel(f"Active Theme: {self.theme_combo.currentText().capitalize()}")
        self.theme_preview_label.setObjectName("subtitleLabel")

        self.theme_swatch = QLabel()
        self.theme_swatch.setFixedSize(24, 24)
        self.theme_swatch.setStyleSheet("border: 1px solid #9bb0cf; border-radius: 4px;")
        self._update_theme_swatch(self.theme_combo.currentText())

        swatch_row = QWidget()
        swatch_layout = QHBoxLayout(swatch_row)
        swatch_layout.setContentsMargins(0, 0, 0, 0)
        swatch_layout.setSpacing(8)
        swatch_layout.addWidget(self.theme_swatch)
        swatch_layout.addWidget(self.theme_preview_label)

        theme_widget = QWidget()
        theme_widget_layout = QVBoxLayout(theme_widget)
        theme_widget_layout.setContentsMargins(0, 0, 0, 0)
        theme_widget_layout.addWidget(self.theme_combo)
        theme_widget_layout.addWidget(swatch_row)

        form_layout.addRow("Camera ID:", self.camera_id_input)
        form_layout.addRow("Refresh Interval (ms):", self.refresh_rate_input)
        form_layout.addRow("Export Directory:", export_dir_widget)
        form_layout.addRow("Theme:", theme_widget)

        action_layout = QHBoxLayout()
        save_button = QPushButton("Apply Settings")
        save_button.clicked.connect(self.save_settings)
        reset_button = QPushButton("Reset Settings")
        reset_button.clicked.connect(self.reset_settings)
        action_layout.addWidget(save_button)
        action_layout.addWidget(reset_button)

        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addLayout(form_layout)
        layout.addLayout(action_layout)

        return container

    def _load_profile_image(self):
        image_path = os.path.join(os.getcwd(), self.settings["ui"]["profile_image"])
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                self.profile_image.setPixmap(pixmap.scaled(160, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                return

        placeholder = QPixmap(160, 160)
        placeholder.fill(Qt.transparent)
        self.profile_image.setPixmap(placeholder)

    def toggle_camera(self):
        if self.camera_running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        if not self.camera_manager.start():
            QMessageBox.warning(self, "Camera Error", "Unable to start the webcam. Please check your camera device.")
            return

        self.camera_timer.start(self.settings["ui"].get("refresh_interval_ms", 30))
        self.camera_running = True
        self.start_button.setText("Stop Camera")
        self.load_history()

    def stop_camera(self):
        self.camera_timer.stop()
        self.camera_manager.stop()
        self.camera_running = False
        self.start_button.setText("Start Camera")

    def update_frame(self):
        success, frame = self.camera_manager.read_frame()
        if not success or frame is None:
            return

        analysis = self.analysis_engine.analyze(frame)
        self.last_summary = analysis["summary"]
        inserted = self._should_insert_session(analysis["summary"])
        if inserted:
            self.database_manager.insert_session(analysis["summary"])
            self.load_history()

        self.display_frame(frame, analysis["faces"])
        self.update_summary(analysis["summary"])

    def display_frame(self, frame, faces):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for face in faces:
            x, y, w, h = face["box"]
            emotion = face["emotion"]
            age = face["age"]
            gender = face["gender"]
            label = f"{emotion} | {age} | {gender}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 180, 255), 2)
            cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        height, width, channels = frame.shape
        bytes_per_line = channels * width
        qt_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(pixmap)

    def update_summary(self, summary):
        self.emotion_value.setText(summary.get("emotion", "Unknown"))
        self.age_value.setText(str(summary.get("age", "N/A")))
        self.gender_value.setText(summary.get("gender", "Unknown"))
        self.attention_value.setText(summary.get("attention", "N/A"))
        self.stress_value.setText(summary.get("stress", "N/A"))
        head_pose_text = summary.get("head_pose", "Center")
        self.head_pose_value.setText(head_pose_text)
        self.head_pose_icon.setStyleSheet(
            f"border-radius: 7px; background: {self._head_pose_color(head_pose_text)};"
        )
        self.face_count_value.setText(str(summary.get("face_count", 0)))

        self.history_label.setText(
            f"Emotion: {summary.get('emotion', 'Unknown')} | Age: {summary.get('age', 'N/A')} | Gender: {summary.get('gender', 'Unknown')} | Head Pose: {summary.get('head_pose', 'Center')} | Faces: {summary.get('face_count', 0)}"
        )

    def _head_pose_color(self, head_pose_text):
        color_map = {
            "Center": "#42c26a",
            "Left": "#f1c40f",
            "Right": "#f1c40f",
        }
        return color_map.get(head_pose_text, "#9bb0cf")

    def load_history(self):
        history = self.database_manager.fetch_history(limit=8)
        self.history_list.clear()
        for row in history:
            timestamp, emotion, age, gender, attention, stress, head_pose, face_count = row
            item = QListWidgetItem(
                f"{timestamp[:19]} — {emotion} | {age} | {gender} | Head Pose: {head_pose} | Faces: {face_count}"
            )
            self.history_list.addItem(item)

    def capture_image(self):
        if self.video_label.pixmap() is None:
            QMessageBox.information(self, "Capture Image", "No camera frame is available to capture.")
            return

        export_dir = os.path.join(self._get_export_dir_path(), "screenshots")
        os.makedirs(export_dir, exist_ok=True)
        file_name = os.path.join(export_dir, f"capture_{cv2.getTickCount()}.png")
        self.video_label.pixmap().save(file_name, "PNG")
        QMessageBox.information(self, "Capture Image", f"Image saved to {file_name}")

    def export_report(self):
        if self.last_summary is None:
            QMessageBox.information(self, "Export Report", "No session summary available yet. Start the camera first.")
            return

        csv_path = self.report_generator.save_summary_csv(self.last_summary)
        excel_path = self.report_generator.save_summary_excel(self.last_summary)
        pdf_path = self.report_generator.save_summary_pdf(self.last_summary)

        chart_dir = os.path.join(self._get_export_dir_path(), "charts")
        os.makedirs(chart_dir, exist_ok=True)
        pie_path = save_emotion_pie(self.database_manager.fetch_history(limit=50), os.path.join(chart_dir, "emotion_distribution.png"))
        age_path = save_age_histogram(self.database_manager.fetch_history(limit=50), os.path.join(chart_dir, "age_distribution.png"))

        self._set_chart_preview(pie_path, self.history_chart_pie, "Emotion chart preview")
        self._set_chart_preview(age_path, self.history_chart_age, "Age histogram preview")
        self.load_history()
        self.refresh_export_list()
        self.load_analytics()

        self.reports_label.setText(
            f"Latest export bundle:\nCSV: {csv_path}\nExcel: {excel_path}\nPDF: {pdf_path}\nCharts: {pie_path}, {age_path}"
        )

        QMessageBox.information(
            self,
            "Export Report",
            f"Report export completed:\nCSV: {csv_path}\nExcel: {excel_path}\nPDF: {pdf_path}\nCharts: {pie_path}, {age_path}",
        )

    def _set_chart_preview(self, image_path, target_label, fallback_text="Chart preview unavailable."):
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                target_label.setPixmap(pixmap.scaled(target_label.width(), target_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                return
        target_label.setText(fallback_text)

    def load_analytics(self):
        stats = self.database_manager.fetch_statistics()
        self.analytics_summary_label.setText(
            f"Sessions: {stats['total_sessions']} | Avg faces: {stats['average_faces']} | Avg age: {stats['average_age']}\n"
            f"Emotion: {stats['emotion_counts']} | Gender: {stats['gender_counts']} | Head Pose: {stats['head_pose_counts']}"
        )

        chart_dir = os.path.join(self._get_export_dir_path(), "analytics")
        os.makedirs(chart_dir, exist_ok=True)
        history = self.database_manager.fetch_history(limit=200)
        pie_path = save_emotion_pie(history, os.path.join(chart_dir, "analytics_emotion_distribution.png"))
        age_path = save_age_histogram(history, os.path.join(chart_dir, "analytics_age_distribution.png"))
        pose_path = save_head_pose_pie(history, os.path.join(chart_dir, "analytics_head_pose_distribution.png"))
        self._set_chart_preview(pie_path, self.analytics_chart_pie, "Emotion distribution")
        self._set_chart_preview(age_path, self.analytics_chart_age, "Age distribution")
        self._set_chart_preview(pose_path, self.analytics_chart_pose, "Head pose distribution")

    def open_export_folder(self):
        export_folder = self._get_export_dir_path()
        os.makedirs(export_folder, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(export_folder))

    def refresh_export_list(self):
        export_folder = self._get_export_dir_path()
        if not os.path.isdir(export_folder):
            self.export_files_label.setText("No export folder found yet.")
            return

        files = sorted(os.listdir(export_folder), reverse=True)
        if not files:
            self.export_files_label.setText("No export files available.")
            return

        preview = "\n".join(files[:8])
        self.export_files_label.setText(f"Latest exported files:\n{preview}")

    def load_saved_charts(self):
        chart_dir = os.path.join(self._get_export_dir_path(), "charts")
        pie_path = os.path.join(chart_dir, "emotion_distribution.png")
        age_path = os.path.join(chart_dir, "age_distribution.png")
        self._set_chart_preview(pie_path, self.history_chart_pie, "Emotion chart preview")
        self._set_chart_preview(age_path, self.history_chart_age, "Age histogram preview")

    def _apply_theme(self, theme_name):
        theme_map = {
            "dark": "gui/styles.qss",
            "light": "gui/styles_light.qss",
        }
        theme_path = theme_map.get(theme_name, "gui/styles.qss")
        if os.path.exists(theme_path):
            with open(theme_path, "r", encoding="utf-8") as style_file:
                self.setStyleSheet(style_file.read())
        else:
            self.setStyleSheet("")

    def _get_export_dir_path(self):
        export_dir = self.settings["analytics"].get("export_dir", "exports")
        if os.path.isabs(export_dir):
            return export_dir
        return os.path.abspath(export_dir)

    def _should_insert_session(self, summary):
        if self.last_saved_summary is None:
            self.last_saved_summary = summary.copy()
            return True

        if (
            summary.get("emotion") != self.last_saved_summary.get("emotion")
            or summary.get("attention") != self.last_saved_summary.get("attention")
            or summary.get("stress") != self.last_saved_summary.get("stress")
            or summary.get("head_pose") != self.last_saved_summary.get("head_pose")
            or summary.get("face_count") != self.last_saved_summary.get("face_count")
        ):
            self.last_saved_summary = summary.copy()
            return True

        return False

    def update_theme_preview(self, theme_name):
        self.theme_preview_label.setText(f"Active Theme: {theme_name.capitalize()}")
        self._update_theme_swatch(theme_name)

    def _update_theme_swatch(self, theme_name):
        swatch_map = {
            "dark": "background: #69c0ff; border: 1px solid #2859df;",
            "light": "background: #87a1d5; border: 1px solid #9bb0cf;",
        }
        self.theme_swatch.setStyleSheet(
            f"border-radius: 4px; {swatch_map.get(theme_name, 'background: #d8e2f1; border: 1px solid #9bb0cf;')}"
        )

    def browse_export_folder(self):
        selected_folder = QFileDialog.getExistingDirectory(self, "Select Export Folder", self._get_export_dir_path())
        if selected_folder:
            self.export_dir_input.setText(selected_folder)

    def save_settings(self):
        self.settings["app"]["camera_id"] = int(self.camera_id_input.value())
        self.settings["ui"]["refresh_interval_ms"] = int(self.refresh_rate_input.value())
        selected_export = self.export_dir_input.text().strip()
        if selected_export:
            if selected_export.startswith(os.getcwd()):
                export_dir_value = os.path.relpath(selected_export, os.getcwd())
            else:
                export_dir_value = selected_export
            self.settings["analytics"]["export_dir"] = export_dir_value
            self.report_generator.export_dir = Path(self.settings["analytics"]["export_dir"])
            self.report_generator.export_dir.mkdir(parents=True, exist_ok=True)
        self.settings["app"]["theme"] = self.theme_combo.currentText()
        self._apply_theme(self.settings["app"]["theme"])
        self.camera_manager.device_index = int(self.settings["app"]["camera_id"])
        self._save_settings_file()
        self.statusBar().showMessage("Settings updated and saved.", 5000)
        QMessageBox.information(self, "Settings Updated", "Settings have been saved to configuration and applied for this session.")

    def _save_settings_file(self):
        try:
            config_dir = os.path.dirname(self.config_path)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as config_file:
                yaml.safe_dump(self.settings, config_file, sort_keys=False)
        except Exception as exc:
            QMessageBox.warning(self, "Save Error", f"Unable to save settings: {exc}")

    def reset_settings(self):
        self.camera_id_input.setValue(self.settings["app"].get("camera_id", 0))
        self.refresh_rate_input.setValue(self.settings["ui"].get("refresh_interval_ms", 30))
        self.export_dir_input.setText(self._get_export_dir_path())
        self.theme_combo.setCurrentText(self.settings["app"].get("theme", "dark"))
        self.statusBar().showMessage("Settings reset to saved configuration.", 3000)

    def closeEvent(self, event):
        self.stop_camera()
        self.database_manager.close()
        super().closeEvent(event)
