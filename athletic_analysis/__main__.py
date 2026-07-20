import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from athletic_analysis.ui import theme
    from athletic_analysis.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Athletic Analysis")
    theme.apply(app)
    window = MainWindow()
    window.show()
    # Allow opening a video from the command line: athlete-analysis clip.mp4
    if len(sys.argv) > 1:
        window.open_video(sys.argv[1])
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
