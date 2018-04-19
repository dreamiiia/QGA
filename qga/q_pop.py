# -*- coding: utf-8 -*-
"""
@author: dream01
"""
import numpy as np
from qga.q_gate import HGate, StaticRotGate, DynamicRotGate_GL, DynamicRotGate_GE, DynamicRotGate_F
import matplotlib.pyplot as plt


QuBitZero = np.array([1,0])
init_qubit = np.dot(HGate, QuBitZero)


class QuantumPopulation(object):
    """
    量子种群
    """
    def __init__(self, question, popSize=15, max_generation=150, step=0.025*np.pi, max_=True):
        self.question = question
        self.step = step                                                                    # 静态旋转角步长

        self.generation = 0                                                                 # 量子种群当前代数
        self.chromosomeLength = self.question.m                                             # 染色体长度，由问题决定
        self.max_generation = max_generation                                                # 设置进化的最大代数
        self.popSize = popSize                                                              # 设置种群大小

        self.quantum_pop = np.empty([self.popSize, 2, self.chromosomeLength])               # 量子种群 quantum population
        self.square_quantum_pop = np.empty([self.popSize, 2, self.chromosomeLength])        # 平方处理后的量子种群 quantum population
        self.pop = np.empty([self.popSize, self.chromosomeLength], dtype=np.int)            # 量子种群经过观测后的种群 population
        self.pop_fitness = np.empty([self.popSize])                                         # 量子种群每个染色体的适应度
        self.pop_solution = [None for _ in range(self.popSize)]                             # 量子种群每个染色体解码之后的表示，对于多变量问题，解码之后可能是个元组，所以使用Python内置的list

        self.best_fitness = None                                                            # 全局最优适应度，与问题有关
        self.best_chrom = None                                                              # 全局最优染色体（二进制码串）
        self.best_solution = None                                                           # 全局最优解（解码之后的数）

        self.first_better_fitness_generation = None                                         # 第一次获得较优解的代数

        self.max = max_

        self.best_fitness_of_each_generation = np.empty(self.max_generation + 1)            # 记录每一代最优适应度，与问题有关
        self.avg_fitness_of_each_generation = np.empty(self.max_generation + 1)             # 记录每一代平均适应度，与问题有关
        self.median_fitness_of_each_generation = np.empty(self.max_generation + 1)          # 记录每一代中位适应度，与问题有关
        self.best_global_fitness_of_each_generation = np.empty(self.max_generation + 1)     # 每一代累积的最佳

        self.fitness_var_of_each_generation = np.empty(self.max_generation + 1)             # 每一代适应度的方差
        self.fitness_std_of_each_generation = np.empty(self.max_generation + 1)             # 每一代适应度的标准差
        self.better_fitness_count_of_each_generation = np.empty(self.max_generation + 1)                       # 每一代获得较优解的次数 0~0.01
        # self.good_fitness_count_of_each_generation = np.empty(self.max_generation + 1)                       # 每一代获得优解的次数   0.01~0.1

        self.r_theta = np.empty([self.popSize, self.chromosomeLength])

    def init_population(self):
        # print(self.quantum_pop)     # 初始化之前
        for i in range(self.popSize):
            for j in range(self.chromosomeLength):
                self.quantum_pop[i][:, j] = init_qubit
                self.square_quantum_pop[i][:, j] = [0.5, 0.5]
        # print(self.quantum_pop)     # 初始化之后
        # print(self.square_quantum_pop)

    def show_population_infor(self):
        print("量子种群\n", self.quantum_pop)       # **
        print("平方后的量子种群\n", self.square_quantum_pop)    # **
        print("观测种群\n", self.pop)   # **
        print("最优染色体信息，染色体组成{}，适应度{},解码解{}".format(self.best_chrom, self.best_fitness, self.best_solution))

        print("当前代数：", self.generation)
        print("当前代数的平均适应度", self.avg_fitness_of_each_generation[self.generation])
        print("当前代数的中位适应度", self.median_fitness_of_each_generation[self.generation])
        print("当前代数的最优适应度", self.best_fitness_of_each_generation[self.generation])
        print("当前代数的适应度方差", self.fitness_var_of_each_generation[self.generation])
        print("当前代数的适应度标准差", self.fitness_std_of_each_generation[self.generation])
        print("当前代数取得较优解的次数{}，总共有{}个解(种群数量)".format(self.better_fitness_count_of_each_generation[self.generation], self.popSize))
        # print(self.r_theta/np.pi)
        # print(np.max(self.pop_fitness), np.mean(self.pop_fitness))
        # for i, v in enumerate(self.pop_fitness):
        #     print("第{}个染色体的适应度为{}".format(i,v))

    def measure(self):
        """
        对量子种群进行一次观测
        """
        for i in range(self.popSize):
            for j in range(self.chromosomeLength):
                qubit = self.quantum_pop[i][:, j]
                square_qubit = self.square_quantum_pop[i][:, j]
                random_num = np.random.uniform(0, 1)
                # if np.square(qubit[0]) >= random_num:       # alpha 的平方代表着观测为0的概率
                if square_qubit[0] >= random_num:       # alpha 的平方代表着观测为0的概率
                    self.pop[i, j] = 0
                else:
                    self.pop[i, j] = 1
                # print(i, j, square_qubit[0], np.square(qubit[0]), random_num, self.pop[i, j])

    def evaluate_chrom(self, chrom):
        """
        对传入的二进制串（染色体）进行评估
        :param chrom: 二进制串
        :return: 二进制解码之后的表示和适应度，对于连续函数优化问题就是问题的解
        """
        return self.question.get_fargs_and_result(chrom)

    def evaluate_population(self):
        """
        对量子种群进行评估
        """
        for i, chrom in enumerate(self.pop):
            self.pop_solution[i], self.pop_fitness[i] = self.evaluate_chrom(chrom)

        if self.max:
            best_fitness = np.max(self.pop_fitness)                            # 当代种群最优(最大)适应度
        else:
            best_fitness = np.min(self.pop_fitness)                            # 当代种群最优(最小)适应度

        # 找寻首次找到较优解的代数
        if not self.first_better_fitness_generation and best_fitness >= self.question.better:
            self.first_better_fitness_generation = self.generation

        best_chrom_index, = np.where(self.pop_fitness == best_fitness)         # 当代种群的最优适应度个体的位置
        best_chrom_index = best_chrom_index[0]

        avg_fitness = np.average(self.pop_fitness)                             # 当代种群的平均适应度
        median_fitness = np.median(self.pop_fitness)                           # 当代种群适应度的中位数
        variance = np.var(self.pop_fitness)                                    # 当代种群适应度方差
        std = np.std(self.pop_fitness)                                         # 当代种群适应度标准差
        better_fitness_count = np.sum(self.pop_fitness >= self.question.better)

        # good_fitness_count = np.sum(np.logical_and(self.pop_fitness < self.question.better,
        #                                            self.question.good <= self.pop_fitness))

        # print('第{}代种群中最优适应度为：{}，二进制表示为：{}，解码值为：{}'.format(self.generation, best_fitness,
        #                                                   self.pop[best_chrom_index], self.pop_solution[best_chrom_index]))
        self.best_fitness_of_each_generation[self.generation] = best_fitness
        self.avg_fitness_of_each_generation[self.generation] = avg_fitness
        self.median_fitness_of_each_generation[self.generation] = median_fitness
        self.update_best_fitness(best_chrom_index)                             # 更新全局最佳个体
        self.best_global_fitness_of_each_generation[self.generation] = self.best_fitness

        self.fitness_var_of_each_generation[self.generation] = variance
        self.fitness_std_of_each_generation[self.generation] = std
        self.better_fitness_count_of_each_generation[self.generation] = better_fitness_count
        # self.good_fitness_count_of_each_generation[self.generation] = good_fitness_count


    def update_best_fitness(self, best_chrom_index):
        """
        用于更新全局最佳个体
        """
        current_best_fitness = self.pop_fitness[best_chrom_index]
        current_best_chrom = self.pop[best_chrom_index].copy()
        def has_better_fitness():
            if self.max:
                return current_best_fitness > self.best_fitness
            else:
                return current_best_fitness < self.best_fitness
        if self.best_fitness is None or has_better_fitness():
            self.best_fitness = current_best_fitness                    # 更新全局最优适应度
            self.best_chrom = current_best_chrom                        # 更新全局最优适应度的观测后的染色体组成
            self.best_solution = self.pop_solution[best_chrom_index]    # 更新全局最优适应度的染色体解码值
            print("当前代数为：{}，在种群位置为:{}, 找到一个更好的解：{}啦，它的经典染色体组成：{}，解码后的值为{}，".format(  # ***
                self.generation, best_chrom_index, self.best_fitness, self.best_chrom, self.best_solution))                    # ***

    def update_quantum_population(self):
        """
        对量子种群进行旋转更新
        """
        for i, quantum_chrom in enumerate(self.quantum_pop):
            for j in range(self.chromosomeLength):
                qubit = quantum_chrom[:, j]  # 获得量子比特
                current_fitness = self.pop_fitness[i]  # 当前解的适应度
                x_i = self.pop[i][j]                # 当前染色体第i位的值
                best_i = self.best_chrom[j]         # 全局最优解染色体第i位值

                direction_kargs = dict(qubit_i=qubit, x_i=x_i, best_i=best_i)
                # r = StaticRotGate(**direction_kargs, step=self.step)                          # 单值静态旋转角幅值的策略
                # r = DynamicRotGate_GL(**direction_kargs,                                        # 基于遗传代数的动态旋转角策略(线性)
                #                       g=self.generation, g_max=self.max_generation)
                # r = DynamicRotGate_GE(**direction_kargs,                                        # 基于遗传代数的动态旋转角策略（指数）
                #                       g=self.generation, g_max=self.max_generation)

                r = DynamicRotGate_F(**direction_kargs,                                         # 基于适应度的动态策略
                                     f_current=current_fitness, f_best=self.best_fitness)

                self.r_theta[i][j] = r.get_r_theta()
                r_gate = r.rot_gate

                # 更新量子种群
                quantum_chrom[:, j] = np.around(np.dot(r_gate, qubit), 2)

                #  更新平方后的量子种群
                self.square_quantum_pop[i, 0, j] = np.around(np.square(quantum_chrom[0, j]), 2)
                self.square_quantum_pop[i, 1, j] = 1-self.square_quantum_pop[i, 0, j]

        print("更新角度策略：", self.r_theta/np.pi)  # **
        # print('新的量子种群:', self.quantum_pop)

    def draw_pictual_1(self):
        """
        进行画图，横坐标是遗传代数generation，纵坐标为适应度fitness，两条曲线，平均适应度和最优适应度
        都在一幅图中
        """
        x = np.arange(0, self.max_generation+1, dtype=np.int)
        # plt.subplot(2, 1, 1)        # 3行1列，第1个位置
        plt.plot(x, self.best_fitness_of_each_generation, 'r', label="best_fitness")
        plt.plot(x, self.best_global_fitness_of_each_generation, 'y', label="accu_fitness")
        # plt.legend()  # 展示图例
        # plt.subplot(2, 2, 2)
        plt.plot(x, self.avg_fitness_of_each_generation, 'g', label="avg_fitness")
        # plt.legend()  # 展示图例
        # plt.subplot(2, 2, 3)
        plt.plot(x, self.median_fitness_of_each_generation, 'b', label="median_fitness")
        plt.legend()  # 展示图例

        plt.xlabel('Generation')            # 给 x 轴添加标签
        plt.ylabel('Fitness')               # 给 y 轴添加标签
        plt.title('Fitness and Generation')      # 添加图形标题
        plt.show()

    def draw_pictual_2(self):
        x = np.arange(0, self.max_generation + 1, dtype=np.int)
        plt.subplot(3, 1, 1)        # 3行1列，第1个位置
        plt.plot(x, self.best_fitness_of_each_generation, 'r', label="best_fitness")
        # plt.plot(x, self.best_global_fitness_of_each_generation, 'y', label="accu_fitness")
        plt.legend()  # 展示图例
        plt.subplot(3, 1, 2)
        plt.plot(x, self.avg_fitness_of_each_generation, 'g', label="avg_fitness")
        plt.legend()  # 展示图例
        plt.subplot(3, 1, 3)
        plt.plot(x, self.median_fitness_of_each_generation, 'b', label="median_fitness")
        plt.legend()  # 展示图例

        # plt.subplot(3, 2, 4)
        # plt.plot(x, self.fitness_var_of_each_generation, 'b', label="variance")
        # plt.legend()  # 展示图例
        #
        # plt.subplot(3, 2, 5)
        # plt.plot(x, self.fitness_std_of_each_generation, 'b', label="std")
        # plt.legend()  # 展示图例
        #
        # plt.subplot(3, 2, 6)
        # plt.plot(x, self.better_fitness_count_of_each_generation, 'b', label="better_fitness_count")
        # plt.legend()  # 展示图例

        # plt.xlabel('Generation')  # 给 x 轴添加标签
        # plt.ylabel('Fitness')  # 给 y 轴添加标签
        # plt.title('Fitness and Generation')  # 添加图形标题
        plt.show()
