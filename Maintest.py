import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import uic
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import math
import random
import time
from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QVBoxLayout, QLineEdit, QLabel, QGroupBox, QPushButton, QApplication, QWidget
from PyQt5.uic import loadUi









e=math.e
DNA_SIZE = 24
POP_SIZE = 100
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.01
MUTATION_AMOUNT=0.05
N_GENERATIONS = 48
CHECKTIME = 20 #独立验证次数

#POP_SIZE = 100
#CROSSOVER_RATE = 0.8
#MUTATION_RATE = 0.01
#MUTATION_AMOUNT=0.05
#N_GENERATIONS = 48
#CHECKTIME = 20 #独立验证次数






ELITE_RATE=0.50
X_BOUND = [-4, 4]
Y_BOUND = [-4, 4]
CATASTROPHE_RATE = 0.001
#比较参数：POP_SIZE = 400，POP_SIZE = 400（精算）；POP_SIZE = 200,N_GENERATIONS = 48（速算）

def GA(q):
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
    x_pattern1=x_axis_for_2d
    y_pattern1_matrix=pop_matrix


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
    x_pattern0 = x
    y_pattern0 = each_y
    if (q == 0):#收敛值分析
        return x_pattern0,y_pattern0#决定最后输出什么类型的图表
    elif(q == 1) :#收敛速度分析
        return  x_pattern1,y_pattern1_matrix
    else:
        print("问题在这里")
        print(q)
def GA_1(q):
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
    x_pattern1 = x_axis_for_2d
    y_pattern1 = pop_matrix



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
    x_pattern0 = x
    y_pattern0 = each_y
    print(q)
    if(q == 1):
        return x_pattern1,y_pattern1
    elif(q==0):
        return x_pattern0,y_pattern0
def AGA(q):
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
    def crossover_and_mutation(fitness,pop):
        new_pop = []
        count = 0
        fitness_max=max(fitness)
        fitness_avg=sum(fitness)/POP_SIZE
        var = np.var(fitness)
        # 计算种群稠密度
        density = 1 / (1 + math.exp(-var))
        MUTATION_RATE_FACTOR = (1 - ((density - 0.5) / 0.5)) * 0.01  # 反向归一化，再映射到0~0.1作为变异率
        CROSSOVER_RATE_FACTOR = (1 - ((density - 0.5) / 0.5)) * 0.5 +0.4 # 反向归一化，再映射到0.4~0.9作为交叉率
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
                CROSSOVER_RATE_dynamic=(CROSSOVER_RATE*((fitness_max-f)/(fitness_max-fitness_avg)))*0.5+(CROSSOVER_RATE_FACTOR)*0.5
                # CROSSOVER_RATE_dynamic=CROSSOVER_RATE*((fitness_max-f)/(fitness_max-fitness_avg))
                #动态规划,好个体减少交叉率和变异率
                # print("这是比较好的个体的交叉率")
                # print(CROSSOVER_RATE_dynamic)
                MUTATION_RATE_dynamic=(MUTATION_RATE*((fitness_max-f)/(fitness_max-fitness_avg)))*0.5+(MUTATION_RATE_FACTOR)*0.5

            elif f<fitness_avg:
                CROSSOVER_RATE_dynamic=CROSSOVER_RATE_FACTOR

                MUTATION_RATE_dynamic=MUTATION_RATE_FACTOR

            if np.random.rand() < CROSSOVER_RATE_dynamic:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
                cross_points = np.random.randint(low=0, high=DNA_SIZE * 2)  # 随机产生交叉的点
                child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
            if np.random.rand() < MUTATION_RATE_dynamic:
                mutation(child)  # 每个后代有一定的机率发生变异
            new_pop.append(child)

        return new_pop
    def mutation(child):
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

        return pop[idx]
    def print_info(pop):
        fitness = get_fitness(pop)
        # print(fitness)
        max_fitness_index = np.argmax(fitness)
        print("max_fitness:", fitness[max_fitness_index])

        x, y = translateDNA(pop)
        print("最优的基因型：", pop[max_fitness_index])
        print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))
    def get_var(pop):
        fitness = get_fitness(pop)
        var=np.var(fitness)
        #计算种群稠密度,映射到1~0.5，越接近1越稀疏
        density =1/(1+math.exp(-var))
        MUTATION_RATE_FACTOR = (1-((density - 0.5)/0.5))*0.001 #反向归一化，再映射到0~0.001作为变异率
        print(MUTATION_RATE_FACTOR)
        # 一般来说，到最后可能整个区域都是一个基因型，需要变动
        # print(density)
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
            pop = np.array(crossover_and_mutation(fitness, pop))
            pop = select(pop, fitness)  # 选择生成新的种群
            temp.append(F(x[max_fitness_index], y[max_fitness_index]))
            # get_var(pop)

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
    x_pattern1=x_axis_for_2d
    y_pattern1_matrix=pop_matrix


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
    x_pattern0 = x
    y_pattern0 = each_y
    if (q == 0):#收敛值分析
        return x_pattern0,y_pattern0#决定最后输出什么类型的图表
    elif(q == 1) :#收敛速度分析
        return  x_pattern1,y_pattern1_matrix
    else:
        print("问题在这里")
        print(q)



import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QGroupBox
from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.uic import loadUi




#
#
# POP_SIZE = 100
# CROSSOVER_RATE = 0.8
# MUTATION_RATE = 0.01
# MUTATION_AMOUNT=0.05
# N_GENERATIONS = 48
# CHECKTIME = 20 #独立验证次数



class GeneticAlgorithmUI(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("遗传算法界面")
        self.initUI()

    def initUI(self):
        uic.loadUi("UI.ui", self)
        #触发器

        self.pushButton_17.clicked.connect(self.submit_parameters)
        self.pushButton_16.clicked.connect(lambda: self.run_algorithm("standard_convergence", 0))
        self.pushButton_14.clicked.connect(lambda: self.run_algorithm("adaptive_convergence", 0))
        self.pushButton_13.clicked.connect(lambda: self.run_algorithm("standard_speed", 1))
        self.pushButton_15.clicked.connect(lambda: self.run_algorithm("adaptive_speed", 1))

    def submit_parameters(self):
        # 根据控件名称获取对应的控件对象
        global POP_SIZE, CROSSOVER_RATE, MUTATION_RATE, MUTATION_AMOUNT, N_GENERATIONS, CHECKTIM
        POP_SIZE= int(self.textEdit_13.toPlainText())
        CROSSOVER_RATE = float(self.textEdit_14.toPlainText())
        MUTATION_RATE = float(self.textEdit_15.toPlainText())
        MUTATION_AMOUNT =float( self.textEdit_16.toPlainText())
        N_GENERATIONS = int(self.textEdit_17.toPlainText())
        CHECKTIME = int(self.textEdit_18.toPlainText())

        print("目前的参数如下",POP_SIZE, CROSSOVER_RATE, MUTATION_RATE, MUTATION_AMOUNT, N_GENERATIONS, CHECKTIME)

    #     # Algorithm buttons


    def run_algorithm(self, algorithm_type=None, analysis_type=None):
        global POP_SIZE, CROSSOVER_RATE, MUTATION_RATE, MUTATION_AMOUNT, N_GENERATIONS, CHECKTIME
        POP_SIZE = int(self.textEdit_13.toPlainText())
        CROSSOVER_RATE = float(self.textEdit_14.toPlainText())
        MUTATION_RATE = float(self.textEdit_15.toPlainText())
        MUTATION_AMOUNT = float(self.textEdit_16.toPlainText())
        N_GENERATIONS = int(self.textEdit_17.toPlainText())
        CHECKTIME = int(self.textEdit_18.toPlainText())


        print( "跑算法的函数被调用了！")
        if algorithm_type == "standard_convergence":
            # 调用标准遗传算法函数，收敛值分析
            x, y = GA(analysis_type,)
            self.create_standard_image_amount(x, y)
        elif algorithm_type == "adaptive_convergence":
            # 调用自适应遗传算法函数，收敛值分析
            x, y = AGA(analysis_type)
            self.create_adaptive_image_amount(x, y)
        elif algorithm_type == "standard_speed":
            # 调用标准遗传算法函数，收敛速度分析
            x, y = GA(analysis_type)
            self.create_standard_image_speed(x, y)
        elif algorithm_type == "adaptive_speed":
            # 调用自适应遗传算法函数，收敛速度分析
            x, y = AGA(analysis_type)
            self.create_adaptive_image_speed(x, y)

    def display_image(self, layout, image):
        # 清除原有的图片
        self.clear_images(layout)
        # 添加新的图片
        layout.addWidget(image)

    def create_standard_image_amount(self, x, y):
        fig = plt.Figure(figsize=(3.99, 3.89))

        ax = fig.add_subplot(111)
        ax.plot(x, y)
        ax.set_xlabel('实验次数')
        ax.set_ylabel('最终收敛值')
        ax.set_title('独立测试中各种群最终收敛值', fontsize=20)
        ax.set_ylim(0, 1)

        # Create a Canvas object from the figure
        canvas = FigureCanvas(fig)
        # Convert Canvas to QPixmap
        pixmap = QPixmap(canvas.size())
        canvas.render(pixmap)
        # Set the QPixmap to the QLabel
        self.label_22.setPixmap(pixmap)








    def create_standard_image_speed(self, x, y):

        fig = plt.Figure(figsize=(3.99, 3.89))

        ax = fig.add_subplot(111)
        ax.set_ylim(0, 1)
        for i in range(CHECKTIME):
            ax.plot(x, y[i])
            ax.set_xlabel('迭代次数')
            ax.set_ylabel('每世代各种群的收敛值')
            ax.set_title('收敛速度分析', fontsize=20)

        # Create a Canvas object from the figure
        canvas = FigureCanvas(fig)
        # Convert Canvas to QPixmap
        pixmap = QPixmap(canvas.size())
        canvas.render(pixmap)
        # Set the QPixmap to the QLabel
        self.label_22.setPixmap(pixmap)

    def create_adaptive_image_amount(self,x,y):
        fig = plt.Figure(figsize=(3.99, 3.89))

        ax = fig.add_subplot(111)
        ax.plot(x, y)
        ax.set_xlabel('实验次数')
        ax.set_ylabel('最终收敛值')
        ax.set_title('独立测试中各种群最终收敛值', fontsize=20)
        ax.set_ylim(0, 1)

        # Create a Canvas object from the figure
        canvas = FigureCanvas(fig)
        # Convert Canvas to QPixmap
        pixmap = QPixmap(canvas.size())
        canvas.render(pixmap)
        # Set the QPixmap to the QLabel
        self.label_23.setPixmap(pixmap)

    def create_adaptive_image_speed(self,x,y):
        fig = plt.Figure(figsize=(3.99, 3.89))


        ax = fig.add_subplot(111)
        ax.set_ylim(0, 1)
        for i in range(CHECKTIME):
            ax.plot(x, y[i])
            ax.set_xlabel('迭代次数')
            ax.set_ylabel('每世代各种群的收敛值')
            ax.set_title('收敛速度分析', fontsize=20)

            # Create a Canvas object from the figure
            canvas = FigureCanvas(fig)
            # Convert Canvas to QPixmap
            pixmap = QPixmap(canvas.size())
            canvas.render(pixmap)
            # Set the QPixmap to the QLabel
            self.label_23.setPixmap(pixmap)



if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    window = GeneticAlgorithmUI()
    window.show()
    sys.exit(app.exec_())


