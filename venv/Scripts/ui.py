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
        self.mutation_amount_input = QLineEdit()
        self.iterations_input = QLineEdit()
        self.validation_iterations_input = QLineEdit()
        input_layout.addWidget(QLabel("种群大小:"))
        input_layout.addWidget(self.population_size_input)
        input_layout.addWidget(QLabel("初始交叉率:"))
        input_layout.addWidget(self.initial_crossover_rate_input)
        input_layout.addWidget(QLabel("初始变异率:"))
        input_layout.addWidget(self.initial_mutation_rate_input)
        input_layout.addWidget(QLabel("变异程度:"))
        input_layout.addWidget(self.mutation_amount_input)
        input_layout.addWidget(QLabel("迭代次数:"))
        input_layout.addWidget(self.iterations_input)
        input_layout.addWidget(QLabel("独立验证次数:"))
        input_layout.addWidget(self.validation_iterations_input)
        input_group = QGroupBox("输入区域")
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Algorithm buttons
        alg_buttons_layout = QVBoxLayout()

        self.standard_convergence_button = QPushButton("标准遗传算法的收敛值分析")
        self.standard_convergence_button.clicked.connect(lambda: self.run_algorithm("standard_convergence", 0))
        self.adaptive_convergence_button = QPushButton("自适应遗传算法的收敛值分析")
        self.adaptive_convergence_button.clicked.connect(lambda: self.run_algorithm("adaptive_convergence", 0))
        self.standard_speed_button = QPushButton("标准遗传算法的收敛速度分析")
        self.standard_speed_button.clicked.connect(lambda: self.run_algorithm("standard_speed", 1))
        self.adaptive_speed_button = QPushButton("自适应遗传算法的收敛速度分析")
        self.adaptive_speed_button.clicked.connect(lambda: self.run_algorithm("adaptive_speed", 1))
        alg_buttons_layout.addWidget(self.standard_convergence_button)
        alg_buttons_layout.addWidget(self.adaptive_convergence_button)
        alg_buttons_layout.addWidget(self.standard_speed_button)
        alg_buttons_layout.addWidget(self.adaptive_speed_button)

        layout.addLayout(alg_buttons_layout)

        # Output image groups
        self.standard_image_group = QGroupBox("标准遗传算法结果")
        self.standard_image_layout = QVBoxLayout()
        self.standard_image_group.setLayout(self.standard_image_layout)
        layout.addWidget(self.standard_image_group)

        self.adaptive_image_group = QGroupBox("自适应遗传算法结果")
        self.adaptive_image_layout = QVBoxLayout()
        self.adaptive_image_group.setLayout(self.adaptive_image_layout)
        layout.addWidget(self.adaptive_image_group)

        self.setLayout(layout)


