import datetime
import os
from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


class ReportGenerator:
    def __init__(self, settings):
        self.export_dir = Path(settings["analytics"]["export_dir"])
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def _get_timestamp(self):
        return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    def save_summary_csv(self, summary):
        timestamp = self._get_timestamp()
        file_path = self.export_dir / f"session_summary_{timestamp}.csv"
        pd.DataFrame([summary]).to_csv(file_path, index=False)
        return str(file_path)

    def save_summary_excel(self, summary):
        timestamp = self._get_timestamp()
        file_path = self.export_dir / f"session_summary_{timestamp}.xlsx"
        pd.DataFrame([summary]).to_excel(file_path, index=False)
        return str(file_path)

    def save_summary_pdf(self, summary):
        timestamp = self._get_timestamp()
        file_path = self.export_dir / f"session_summary_{timestamp}.pdf"
        c = canvas.Canvas(str(file_path), pagesize=letter)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(40, 740, "FaceSense AI Pro Session Summary")
        c.setFont("Helvetica", 12)
        y = 700
        for key, value in summary.items():
            c.drawString(40, y, f"{key.capitalize()}: {value}")
            y -= 24
        c.save()
        return str(file_path)
