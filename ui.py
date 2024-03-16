import sys

import untitled as untitled
from PyQt5.QtWidgets import QApplication
from PyQt5 import uic
if __name__ =='__main__':
    app = QApplication(sys.argv)
    ui = uic.loadUi(untitled.ui)
    ui.show()
    app.exec()
