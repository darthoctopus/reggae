if __name__ == "__main__":

    import sys

    from matplotlib.backends.qt_compat import QtWidgets
    from .qtconsole import ReggaeDebugWindow

    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    app = ReggaeDebugWindow()
    print = app.print
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()