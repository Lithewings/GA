import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import math
import random
import time



e=math.e
DNA_SIZE = 24
POP_SIZE = 800
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.01
MUTATION_AMOUNT=0.05
N_GENERATIONS = 48
CHECKTIME = 100 #独立验证次数


ELITE_RATE=0.50
X_BOUND = [-4, 4]
Y_BOUND = [-4, 4]
CATASTROPHE_RATE = 0.01
#比较参数：POP_SIZE = 400，POP_SIZE = 400（精算）；POP_SIZE = 200,N_GENERATIONS = 48（速算）

def AGA():
    def F(x_1, x_2):
        n = 2
        a = 20
        b = 0.2
        c = 2 * np.pi
        return -a * e**(-b * (1 / n * (x_1 **2 + x_2 **2))*0.5) - e**(1 / n * (np.cos(c * x_1)+np.cos(c * x_2))) + a + e
    # def F(x_1, x_2):
    #     return x_1**2+x_2**2
    def plot_3d(ax):
        X = np.linspace(*X_BOUND, 100)
        Y = np.linspace(*Y_BOUND, 100)
        X, Y = np.meshgrid(X, Y)
        Z = F(X, Y)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.pause(3)
        plt.show()
    def get_fitness(pop):
        x, y = translateDNA(pop)
        pred = F(x, y)
        #return pred
        #return pred - np.min(pred)+1e-5  # 求最大值时的适应度
        return np.max(pred) - pred + 1e-5  # 求最小值时的适应度，通过这一步fitness的范围为[0, np.max(pred)-np.min(pred)]
    def translateDNA(pop):  # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
        x_pop = pop[:, 0:DNA_SIZE]  # 前DNA_SIZE位表示X
        y_pop = pop[:, DNA_SIZE:]  # 后DNA_SIZE位表示Y

        x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
        y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]

        return x, y
    def crossover_and_mutation(fitness,pop, CROSSOVER_RATE=0.8):
        new_pop = []
        count = 0
        fitness_max=max(fitness)
        fitness_avg=sum(fitness)/POP_SIZE
        # print("适应度最大值为：",fitness_max)
        # print("适应度平均值为：",fitness_avg)
        for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
            child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
            i = np.random.randint(POP_SIZE)
            mother = pop[np.random.randint(POP_SIZE)]  # 再种群中选择另一个个体，并将该个体作为母亲
            #print(fitness[i])   #母亲的适应度
            #print(fitness[count])   #父亲的适应度
            f = max(fitness[i],fitness[count]) #两者较大的一个
            count += 1
            #print(f)
            if f >= fitness_avg:
                CROSSOVER_RATE_dynamic=CROSSOVER_RATE*((fitness_max-f)/(fitness_max-fitness_avg))#动态规划
                MUTATION_RATE_dynamic=MUTATION_RATE*((fitness_max-f)/(fitness_max-fitness_avg))
            elif f<fitness_avg:
                CROSSOVER_RATE_dynamic=CROSSOVER_RATE
                MUTATION_RATE_dynamic=MUTATION_RATE
            if np.random.rand() < CROSSOVER_RATE_dynamic:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
                cross_points = np.random.randint(low=0, high=DNA_SIZE * 2)  # 随机产生交叉的点
                child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
            if np.random.rand() < MUTATION_RATE_dynamic:
                mutation(child)  # 每个后代有一定的机率发生变异
            new_pop.append(child)

        return new_pop
    def mutation(child, MUTATION_RATE=0.003):
        for _ in range(int(MUTATION_AMOUNT*DNA_SIZE)):
            mutate_point = np.random.randint(0, DNA_SIZE)  # 随机产生一个实数，代表要变异基因的位置
            child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转
    def select(pop, fitness):  # nature selection wrt pop's fitness
        p = (fitness) / (fitness.sum())
        idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                               p=p)
        #__________________精英主义
        if np.random.rand() < ELITE_RATE and count<int(0.25*(N_GENERATIONS)):
            for _ in range(int(0.05*POP_SIZE)):
                elite = np.argmax(p)
                #print("精英是：",elite)
                idx[random.randint(0,POP_SIZE-1)]=idx[elite] #将精英随机选中一定比例的样本进行覆盖
        #_______________灾变
        # if np.random.rand() < CATASTROPHE_RATE and int(0.50*(N_GENERATIONS))>count>int(0.25*(N_GENERATIONS)):
        #     for _ in range(int(0.05 * POP_SIZE)):
        #         elite = np.argmax(p)
        #         idx[elite]=idx[random.randint(0,POP_SIZE-1)]
        return pop[idx]
    def print_info(pop):
        fitness = get_fitness(pop)
        max_fitness_index = np.argmax(fitness)
        print("max_fitness:", fitness[max_fitness_index])
        x, y = translateDNA(pop)
        print("最优的基因型：", pop[max_fitness_index])
        print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # plt.ion()  # 将画图模式改为交互模式，程序遇到plt.show不会暂停，而是继续执行
    # plot_3d(ax)
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))  # matrix (POP_SIZE, DNA_SIZE)
    count=0
    #记录每次收敛的最大值：y，辅助以x构成折线图
    #
    each_y=[]
    # 收敛矩阵
    pop_matrix = []
    for i in range(CHECKTIME):
        temp = []
        for j in range(N_GENERATIONS):  # 迭代N代
            #迭代次数计数器：
            #print("计数器现在是",count)
            count = count + 1

            x, y = translateDNA(pop)

            # if 'sca' in locals():
            #     sca.remove()
            # sca = ax.scatter(x, y, F(x, y), c='black', marker='o')
            # plt.show()
            # plt.pause(0.1)
            #________自适应
            fitness = get_fitness(pop)
            max_fitness_index = np.argmax(fitness)
            pop = np.array(crossover_and_mutation(fitness, pop, CROSSOVER_RATE))
            pop = select(pop, fitness)  # 选择生成新的种群
            temp.append(F(x[max_fitness_index], y[max_fitness_index]))
        count=0
        each_y.append(F(x[max_fitness_index], y[max_fitness_index]))
            #print(temp)
        pop_matrix.append(temp)
        print(i+1,":",CHECKTIME)
        pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))
    #print(pop_matrix)
    #time_end = time.time()
    #time_c = time_end - time_start
    #print('耗时', time_c, 's')



    #收敛过程
    x_axis_for_2d = np.arange(0, N_GENERATIONS, 1)
    for i in range(CHECKTIME):
        plt.plot(x_axis_for_2d,pop_matrix[i])

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.xlabel('迭代次数')
    plt.ylabel('每世代各种群的收敛值')
    plt.suptitle('收敛速度分析', fontsize=20)
    #plt.show()

    x = []
    for i in range(CHECKTIME):
        x.append(i)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.ylim(0, 2)
    plt.xlabel('实验次数')
    plt.ylabel('最终收敛值')
    plt.suptitle('独立测试中各种群最终收敛值', fontsize=20)
    plt.plot(x,each_y)
    #plt.show()
    print("函数的平均最小值是",sum(each_y)/len(each_y))
    print("函数最小值中位数为",np.median(each_y))
    print_info(pop)
    # plt.ioff()
    # plot_3d(ax)
    final_x=x
    final_y=each_y
    return final_x,final_y#决定最后输出什么类型的图表

def GA():
    def F(x_1, x_2):
        n = 2
        a = 20
        b = 0.2
        c = 2 * np.pi
        return -a * e ** (-b * (1 / n * (x_1 ** 2 + x_2 ** 2)) * 0.5) - e ** (
                    1 / n * (np.cos(c * x_1) + np.cos(c * x_2))) + a + e
        # def F(x_1, x_2):
        #     return x_1**2+x_2**2

    def plot_3d(ax):
        X = np.linspace(*X_BOUND, 100)
        Y = np.linspace(*Y_BOUND, 100)
        X, Y = np.meshgrid(X, Y)
        Z = F(X, Y)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.pause(3)
        plt.show()

    def get_fitness(pop):
        x, y = translateDNA(pop)
        pred = F(x, y)
        # return pred
        # return pred - np.min(pred)+1e-5  # 求最大值时的适应度
        return np.max(pred) - pred + 1e-5  # 求最小值时的适应度，通过这一步fitness的范围为[0, np.max(pred)-np.min(pred)]

    def translateDNA(pop):  # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
        x_pop = pop[:, 0:DNA_SIZE]  # 前DNA_SIZE位表示X
        y_pop = pop[:, DNA_SIZE:]  # 后DNA_SIZE位表示Y

        x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[
            0]
        y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[
            0]

        return x, y

    def crossover_and_mutation(fitness, pop, CROSSOVER_RATE=0.8):
        new_pop = []
        count = 0
        fitness_max = max(fitness)
        fitness_avg = sum(fitness) / POP_SIZE
        # print("适应度最大值为：",fitness_max)
        # print("适应度平均值为：",fitness_avg)
        for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
            child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
            i = np.random.randint(POP_SIZE)
            mother = pop[np.random.randint(POP_SIZE)]  # 再种群中选择另一个个体，并将该个体作为母亲
            # print(fitness[i])   #母亲的适应度
            # print(fitness[count])   #父亲的适应度
            f = max(fitness[i], fitness[count])  # 两者较大的一个
            count += 1
            # print(f)
            if f >= fitness_avg:
                CROSSOVER_RATE_dynamic = CROSSOVER_RATE * ((fitness_max - f) / (fitness_max - fitness_avg))  # 动态规划
                MUTATION_RATE_dynamic = MUTATION_RATE * ((fitness_max - f) / (fitness_max - fitness_avg))

                CROSSOVER_RATE_dynamic = CROSSOVER_RATE
                MUTATION_RATE_dynamic = MUTATION_RATE
            elif f < fitness_avg:
                CROSSOVER_RATE_dynamic = CROSSOVER_RATE
                MUTATION_RATE_dynamic = MUTATION_RATE

            if np.random.rand() < CROSSOVER_RATE_dynamic:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
                cross_points = np.random.randint(low=0, high=DNA_SIZE * 2)  # 随机产生交叉的点
                child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
            if np.random.rand() < MUTATION_RATE_dynamic:
                mutation(child)  # 每个后代有一定的机率发生变异
            new_pop.append(child)

        return new_pop

    def mutation(child, MUTATION_RATE=0.003):
        for _ in range(int(MUTATION_AMOUNT * DNA_SIZE)):
            mutate_point = np.random.randint(0, DNA_SIZE)  # 随机产生一个实数，代表要变异基因的位置
            child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转

    def select(pop, fitness):  # nature selection wrt pop's fitness
        p = (fitness) / (fitness.sum())
        idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                               p=p)
        # __________________精英主义
        if np.random.rand() < ELITE_RATE and count < int(0.25 * (N_GENERATIONS)):
            for _ in range(int(0.05 * POP_SIZE)):
                elite = np.argmax(p)
                # print("精英是：",elite)
                idx[random.randint(0, POP_SIZE - 1)] = idx[elite]  # 将精英随机选中一定比例的样本进行覆盖
        # _______________灾变
        # if np.random.rand() < CATASTROPHE_RATE and int(0.50*(N_GENERATIONS))>count>int(0.25*(N_GENERATIONS)):
        #     for _ in range(int(0.05 * POP_SIZE)):
        #         elite = np.argmax(p)
        #         idx[elite]=idx[random.randint(0,POP_SIZE-1)]
        return pop[idx]

    def print_info(pop):
        fitness = get_fitness(pop)
        max_fitness_index = np.argmax(fitness)
        print("max_fitness:", fitness[max_fitness_index])
        x, y = translateDNA(pop)
        print("最优的基因型：", pop[max_fitness_index])
        print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # plt.ion()  # 将画图模式改为交互模式，程序遇到plt.show不会暂停，而是继续执行
        # plot_3d(ax)

    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))  # matrix (POP_SIZE, DNA_SIZE)
    count = 0
    # 记录每次收敛的最大值：y，辅助以x构成折线图
    #
    each_y = []
    # 收敛矩阵
    pop_matrix = []
    for i in range(CHECKTIME):
        temp = []
        for j in range(N_GENERATIONS):  # 迭代N代
            # 迭代次数计数器：
            # print("计数器现在是",count)
            count = count + 1

            x, y = translateDNA(pop)

            # if 'sca' in locals():
            #     sca.remove()
            # sca = ax.scatter(x, y, F(x, y), c='black', marker='o')
            # plt.show()
            # plt.pause(0.1)
            # ________自适应
            fitness = get_fitness(pop)
            max_fitness_index = np.argmax(fitness)
            pop = np.array(crossover_and_mutation(fitness, pop, CROSSOVER_RATE))
            pop = select(pop, fitness)  # 选择生成新的种群
            temp.append(F(x[max_fitness_index], y[max_fitness_index]))
        count = 0
        each_y.append(F(x[max_fitness_index], y[max_fitness_index]))
        # print(temp)
        pop_matrix.append(temp)
        print(i + 1, ":", CHECKTIME)
        pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))
    # print(pop_matrix)
    #time_end = time.time()
    #time_c = time_end - time_start
    #print('耗时', time_c, 's')

    # 收敛过程

    x_axis_for_2d = np.arange(0, N_GENERATIONS, 1)
    for i in range(CHECKTIME):
        plt.plot(x_axis_for_2d, pop_matrix[i])

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.xlabel('迭代次数')
    plt.ylabel('每世代各种群的收敛值')
    plt.suptitle('收敛速度分析', fontsize=20)
    #plt.show()
    final_x = x_axis_for_2d
    final_y = pop_matrix[i]



    x = []
    for i in range(CHECKTIME):
        x.append(i)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.ylim(0, 2)
    plt.xlabel('实验次数')
    plt.ylabel('最终收敛值')
    plt.suptitle('独立测试中各种群最终收敛值', fontsize=20)
    plt.plot(x, each_y)
    #plt.show()
    print("函数的平均最小值是", sum(each_y) / len(each_y))
    print("函数最小值中位数为", np.median(each_y))
    print_info(pop)
    # plt.ioff()
    # plot_3d(ax)
    i = 0
    while(i == 0):
        return final_x,final_y




import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QGroupBox
from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

POP_SIZE = 800
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.01
MUTATION_AMOUNT = 0.05
N_GENERATIONS = 48
CHECKTIME = 100

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

        # Algorithm buttons and output image areas
        algorithm_layout = QVBoxLayout()

        alg_buttons_layout = QHBoxLayout()
        self.standard_alg_button = QPushButton("标准遗传算法")
        self.standard_alg_button.clicked.connect(lambda: self.run_algorithm("standard"))
        self.adaptive_alg_button = QPushButton("自适应遗传算法")
        self.adaptive_alg_button.clicked.connect(lambda: self.run_algorithm("adaptive"))
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

        self.setLayout(layout)

    def run_algorithm(self, algorithm_type=None):
        global POP_SIZE, CROSSOVER_RATE, MUTATION_RATE, MUTATION_AMOUNT, N_GENERATIONS, CHECKTIME
        POP_SIZE = int(self.population_size_input.text())
        CROSSOVER_RATE = float(self.initial_crossover_rate_input.text())
        MUTATION_RATE = float(self.initial_mutation_rate_input.text())
        MUTATION_AMOUNT = float(self.mutation_amount_input.text())
        N_GENERATIONS = int(self.iterations_input.text())
        CHECKTIME = int(self.validation_iterations_input.text())

        if algorithm_type == "standard":
            # 调用标准遗传算法函数
            x,y=GA()
            self.create_standard_image_speed(x,y)
        elif algorithm_type == "adaptive":
            # 调用自适应遗传算法函数
            x,y=AGA()
            self.create_adaptive_image_amount(x,y)

    def display_image(self, layout, image):
        # 清除原有的图片
        self.clear_images(layout)
        # 添加新的图片
        layout.addWidget(image)

    def clear_images(self, layout):
        # 移除布局中的所有部件
        for i in reversed(range(layout.count())):
            layout_item = layout.itemAt(i)
            if layout_item:
                widget = layout_item.widget()
                if widget:
                    widget.setParent(None)

    def create_standard_image_amount(self, x, y):
        fig = Figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        ax.plot(x, y)
        ax.set_xlabel('实验次数')
        ax.set_ylabel('最终收敛值')
        ax.set_title('独立测试中各种群最终收敛值', fontsize=20)
        ax.set_ylim(0, 1)
        canvas = FigureCanvas(fig)
        self.display_image(self.standard_image_layout, canvas)
    def create_standard_image_speed(self, x, y):
        fig = Figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        ax.plot(x, y)
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('每世代各种群的收敛值')
        ax.set_title('收敛速度分析', fontsize=20)
        ax.set_ylim(0, 1)
        canvas = FigureCanvas(fig)
        self.display_image(self.standard_image_layout, canvas)

    def create_adaptive_image_amount(self,x,y):
        fig = Figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        ax.plot(x, y)
        ax.set_xlabel('实验次数')
        ax.set_ylabel('最终收敛值')
        ax.set_title('独立测试中各种群最终收敛值', fontsize=20)
        ax.set_ylim(0, 1)
        canvas = FigureCanvas(fig)
        self.display_image(self.adaptive_image_layout, canvas)
    def create_adaptive_image_speed(self,x,y):
        fig = Figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        ax.plot(x, y)
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('每世代各种群的收敛值')
        ax.set_title('收敛速度分析', fontsize=20)
        ax.set_ylim(0, 1)
        canvas = FigureCanvas(fig)
        self.display_image(self.adaptive_image_layout, canvas)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GeneticAlgorithmUI()
    window.show()
    sys.exit(app.exec_())


