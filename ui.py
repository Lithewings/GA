from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QVBoxLayout, QLineEdit, QLabel, QGroupBox, QPushButton, QApplication, QWidget
from PyQt5.uic import loadUi

population_size = 0
initial_crossover_rate = 0.0
initial_mutation_rate = 0.0
mutation_amount = 0.0
iterations = 0
validation_iterations = 0




# 全局变量
population_size = 0
initial_crossover_rate = 0.0
initial_mutation_rate = 0.0
mutation_amount = 0.0
iterations = 0
validation_iterations = 0

class MyMainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Load the UI file
        uic.loadUi("UI.ui", self)

        # Connect button signal to slot
        self.pushButton_17.clicked.connect(self.submit_parameters)

    def submit_parameters(self):
        population_size = self.textEdit_13.toPlainText()
        # 在这里，你可以将population_size传递给你的宏定义或进行其他操作
        print("种群大小:", population_size)

    def updateUI(self):
        # 在这里更新界面，比如刷新显示当前参数的标签文本
        pass






if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    import sys
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())