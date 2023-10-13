import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
DNA_SIZE = 24
POP_SIZE = 6
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.1
N_GENERATIONS = 100
X_BOUND = [-2.048, 2.048]
Y_BOUND = [-2.048, 2.048]

# List[]
# for i in range(100):
#     List.append(2**i)
# print(List)
List = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472, 274877906944, 549755813888, 1099511627776, 2199023255552, 4398046511104, 8796093022208, 17592186044416, 35184372088832, 70368744177664, 140737488355328, 281474976710656, 562949953421312, 1125899906842624, 2251799813685248, 4503599627370496, 9007199254740992, 18014398509481984, 36028797018963968, 72057594037927936, 144115188075855872, 288230376151711744, 576460752303423488, 1152921504606846976, 2305843009213693952, 4611686018427387904, 9223372036854775808, 18446744073709551616, 36893488147419103232, 73786976294838206464, 147573952589676412928, 295147905179352825856, 590295810358705651712, 1180591620717411303424, 2361183241434822606848, 4722366482869645213696, 9444732965739290427392, 18889465931478580854784, 37778931862957161709568, 75557863725914323419136, 151115727451828646838272, 302231454903657293676544, 604462909807314587353088, 1208925819614629174706176, 2417851639229258349412352, 4835703278458516698824704, 9671406556917033397649408, 19342813113834066795298816, 38685626227668133590597632, 77371252455336267181195264, 154742504910672534362390528, 309485009821345068724781056, 618970019642690137449562112, 1237940039285380274899124224, 2475880078570760549798248448, 4951760157141521099596496896, 9903520314283042199192993792, 19807040628566084398385987584, 39614081257132168796771975168, 79228162514264337593543950336, 158456325028528675187087900672, 316912650057057350374175801344, 633825300114114700748351602688]
List_new = List[:24]
# 线性归一化
#print(sum(List[:24]))
#print(List[24])

pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))
x_pop = pop[:, 0:DNA_SIZE]  # 前DNA_SIZE位表示X
#print(x_pop)
x = x_pop.dot(List_new) / float(2 ** DNA_SIZE - 1)
#print(x)
x=x_pop.dot(List_new) / 16777215
#print(x)
#float(2 ** DNA_SIZE - 1)是调整参数
#一共这个24位的二进制的转换为十进制的最大值就是2^23-1，对每个数据都除上这个数，就映射到了0到1的区间
#print(x)

y=0

i=1-x*np.sin(x*y)-y*y
# best mother leave her DNA: #这会变得不是很随机
# if np.random.rand() < ACE_RATE:
#     max_fitness_index = np.argmax(fitness)
#     cross_points = np.random.randint(low=0, high=DNA_SIZE * 2)
#     mother = pop[max_fitness_index]
#     child[cross_points:] = mother[cross_points:]


# def crossover_and_mutation(fitness, pop, CROSSOVER_RATE=0.8):
#     new_pop = []
#     cross_points = np.random.randint(low=0, high=DNA_SIZE * 2)  # 随机产生交叉的点
#     max_fitness_index = np.argmax(fitness)
#     fitness_max = fitness[max_fitness_index]
#     fitness_avg = sum(fitness) / POP_SIZE
#     # print("适应度平均值为：",fitness_avg)
#     father_fitness = []
#     for i in range(POP_SIZE):
#         father_fitness.append(get_fitness(pop)[i])
#     # print("所有父亲的适应度为：",father_fitness)
#     random_i = random.randint(0, POP_SIZE - 1)
#     mother = pop[random_i]  # 再种群中选择另一个个体，并将该个体作为母亲
#     mother_fitness = get_fitness(pop)[random_i]
#     count = 0
#     for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
#         child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
#
#         # print(random_i)
#
#         # print("母亲的适应度为：",mother_fitness)
#         # print("父亲的适应度为：",father_fitness[count])
#         f = max(father_fitness[count], mother_fitness)
#         # print("父母中适应度更高的是",f)
#         # print("适应度最大值为",fitness_max)
#         # print(fitness_max-f)
#         count += 1
#
#         # print(f)
#         # print("总体适应度最大值为：",fitness_max)
#         # print("总体平均适应度为：",fitness_avg)
#         #     if f>=fitness_avg:
#         #         CROSSOVER_RATE_dynamic=CROSSOVER_RATE*((fitness_max-f)/(fitness_max-fitness_avg))
#         #     elif f<fitness_avg:
#         #         CROSSOVER_RATE_dynamic=CROSSOVER_RATE
#         #     print("此时交叉率为：", CROSSOVER_RATE_dynamic)
#         #     if(fitness_max-f)<0:
#         #         print("_________________________________________")
#         if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
#             child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
#         mutation(child)  # 每个后代有一定的机率发生变异
#         new_pop.append(child)
#     return new_pop