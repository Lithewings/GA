import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QGroupBox
from PyQt5.QtGui import QPixmap

class GeneticAlgorithmUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("遗传算法界面")
        self.initUI()

    def initUI(self):
        layout = QHBoxLayout()

        # Input area
        input_layout = QVBoxLayout()

        self.population_size_input = QLineEdit()
        self.initial_crossover_rate_input = QLineEdit()
        self.initial_mutation_rate_input = QLineEdit()
        self.mutation_amount_input = QLineEdit()  # New input for mutation amount
        self.iterations_input = QLineEdit()
        self.validation_iterations_input = QLineEdit()
        input_layout.addWidget(QLabel("种群大小:"))
        input_layout.addWidget(self.population_size_input)
        input_layout.addWidget(QLabel("初始交叉率:"))
        input_layout.addWidget(self.initial_crossover_rate_input)
        input_layout.addWidget(QLabel("初始变异率:"))
        input_layout.addWidget(self.initial_mutation_rate_input)
        input_layout.addWidget(QLabel("变异程度:"))  # Label for mutation amount
        input_layout.addWidget(self.mutation_amount_input)  # Input box for mutation amount
        input_layout.addWidget(QLabel("迭代次数:"))
        input_layout.addWidget(self.iterations_input)
        input_layout.addWidget(QLabel("独立验证次数:"))  # Label for validation iterations
        input_layout.addWidget(self.validation_iterations_input)  # Input box for validation iterations
        input_group = QGroupBox("输入区域")
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Algorithm buttons and output image areas
        algorithm_layout = QVBoxLayout()

        alg_buttons_layout = QHBoxLayout()
        self.standard_alg_button = QPushButton("标准遗传算法")
        self.standard_alg_button.clicked.connect(self.run_standard_algorithm)
        self.adaptive_alg_button = QPushButton("自适应遗传算法")
        self.adaptive_alg_button.clicked.connect(self.run_adaptive_algorithm)
        alg_buttons_layout.addWidget(self.standard_alg_button)
        alg_buttons_layout.addWidget(self.adaptive_alg_button)
        algorithm_layout.addLayout(alg_buttons_layout)

        # Output image groups
        self.standard_image_group = QGroupBox("标准遗传算法结果")
        self.standard_image_layout = QVBoxLayout()
        self.standard_image_group.setLayout(self.standard_image_layout)
        algorithm_layout.addWidget(self.standard_image_group)

        self.adaptive_image_group = QGroupBox("自适应遗传算法结果")
        self.adaptive_image_layout = QVBoxLayout()
        self.adaptive_image_group.setLayout(self.adaptive_image_layout)
        algorithm_layout.addWidget(self.adaptive_image_group)

        layout.addLayout(algorithm_layout)

        # Run button
        self.run_button = QPushButton("运行")
        self.run_button.clicked.connect(self.run_algorithm)
        layout.addWidget(self.run_button)

        self.setLayout(layout)

    def run_standard_algorithm(self):

        POP_SIZE = int(self.population_size_input.text())
        CROSSOVER_RATE = float(self.initial_crossover_rate_input.text())
        MUTATION_RATE = float(self.initial_mutation_rate_input.text())
        MUTATION_AMOUNT = float(self.mutation_amount_input.text())  # Get mutation amount
        N_GENERATIONS = int(self.iterations_input.text())
        CHECKTIME = int(self.validation_iterations_input.text())  # Get validation iterations
        # Call your standard genetic algorithm function with parameters
        # Replace this with your actual algorithm implementation
        pass

    def run_adaptive_algorithm(self):
        POP_SIZE = int(self.population_size_input.text())
        CROSSOVER_RATE = float(self.initial_crossover_rate_input.text())
        MUTATION_RATE = float(self.initial_mutation_rate_input.text())
        MUTATION_AMOUNT = float(self.mutation_amount_input.text())  # Get mutation amount
        N_GENERATIONS = int(self.iterations_input.text())
        CHECKTIME = int(self.validation_iterations_input.text())  # Get validation iterations
        # Call your adaptive genetic algorithm function with parameters
        # Replace this with your actual algorithm implementation
        pass

    def run_algorithm(self):
        # Placeholder function to run selected algorithm based on user input
        # Replace this with your actual algorithm running code
        pass

    def display_image_standard(self, image_path):
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(512, 512)  # Scale image to 512x512
        label = QLabel()
        label.setPixmap(pixmap)
        self.standard_image_layout.addWidget(label)

    def display_image_adaptive(self, image_path):
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(512, 512)  # Scale image to 512x512
        label = QLabel()
        label.setPixmap(pixmap)
        self.adaptive_image_layout.addWidget(label)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GeneticAlgorithmUI()
    window.show()
    sys.exit(app.exec_())
