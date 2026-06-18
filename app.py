import os
import sys
import yaml

from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow


def load_settings(config_path="config/settings.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    settings = load_settings()

    app = QApplication(sys.argv)
    main_window = MainWindow(settings, config_path="config/settings.yaml")
    main_window.show()

    sys.exit(app.exec())
