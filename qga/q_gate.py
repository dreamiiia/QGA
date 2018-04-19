# -*- coding: utf-8 -*-
"""
Created on 2018/4/15 13:13

@author: dream01
"""
from random import choice
import numpy as np
import matplotlib.pyplot as plt
from utils import str_pi_format

HGate = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])  # H-gate, Hadamard门


class RotGate(object):
    def __init__(self, *, qubit_i, x_i, best_i):
        # 旋转角方向有关的参数
        self.qubit_i = qubit_i
        self.x_i = x_i
        self.best_i = best_i

        self.direction = self.get_direction(qubit_i=self.qubit_i, x_i=self.x_i, best_i=self.best_i)  # 旋转角方向

    def get_step(self):
        """
        需要子类去具体实现
        :return:
        """
        return 0

    @classmethod
    def get_direction(cls, qubit_i, x_i, best_i):
        """
        量子旋转门调整的方向策略
        :param qubit_i: 量子染色体的第i位量子比特
        :param x_i:  量子染色体观测后生成的二进制码串中的第i位
        :param best_i: 保存的全局最优解的第i位
        :return: 量子旋转门调整的方向， 1为逆时针，-1为顺时针，0为不动
        """
        alpha, beta = qubit_i
        if x_i + best_i != 1:
            return 0
        elif best_i == 1:
            if alpha * beta > 0:  # 1、3象限，则逆时针，取正数，靠近|1>
                return 1
            elif alpha * beta < 0:  # 2、4象限，则顺时针，取负数，靠近|1>
                return -1
            elif alpha == 0:  # 在|1>轴上，不用动
                return 0
            else:  # 在|0>轴上，顺时针逆时针动都可以
                return choice([-1, 1])
        elif best_i == 0:
            if alpha * beta > 0:  # 1、3象限，则顺时针，取负数，靠近|0>
                return -1
            elif alpha * beta < 0:  # 2、4象限，则逆时针，取正数，靠近|0>
                return 1
            elif alpha == 0:  # 在|0>轴上，不用动
                return choice([-1, 1])
            else:  # 在|1>轴上，顺时针逆时针动都可以
                return 0

    def get_r_theta(self):
        theta = self.direction * self.get_step()
        return theta

    @classmethod
    def get_rot_gate(cla, r_theta):
        # print("旋转角度为：{}pi".format(r_theta/np.pi))
        return np.array([[np.cos(r_theta), -np.sin(r_theta)], [np.sin(r_theta), np.cos(r_theta)]])


# 单值静态旋转角
class StaticRotGate(RotGate):
    def __init__(self, *, step, qubit_i, x_i, best_i):
        self.step = step                # 获得单值静态旋转角的振幅

        # # 旋转角方向有关的参数
        # self.qubit_i = qubit_i
        # self.x_i = x_i
        # self.best_i = best_i
        #
        # self.direction = self.get_direction(qubit_i=self.qubit_i, x_i=self.x_i, best_i=self.best_i)     # 旋转角方向
        super().__init__(qubit_i=qubit_i, x_i=x_i, best_i=best_i)
        self.r_theta = self.get_r_theta()
        self.rot_gate = self.get_rot_gate(self.r_theta)     # 获得旋转矩阵

    def get_step(self):
        return self.step


class DynamicRotGate_GL(RotGate):
    """
    基于遗传代数的动态旋转角，线性，1
    """
    def __init__(self, *, qubit_i, x_i, best_i, g, g_max):
        super().__init__(qubit_i=qubit_i, x_i=x_i, best_i=best_i)

        self.theta_min = 0.0025 * np.pi
        self.theta_max = 0.05 * np.pi
        self.g = g
        self.g_max = g_max

        self.r_theta = self.get_r_theta()
        self.rot_gate = self.get_rot_gate(self.r_theta)  # 获得旋转矩阵

    def get_step(self):
        step = self.theta_max-(self.theta_max-self.theta_min)*(self.g/self.g_max)
        return step


class DynamicRotGate_GE(RotGate):
    """
    基于遗传代数的动态旋转角，指数型，Exponential
    """
    def __init__(self, *, qubit_i, x_i, best_i, g, g_max):
        super().__init__(qubit_i=qubit_i, x_i=x_i, best_i=best_i)

        self.C = 0.05 * np.pi
        self.g = g
        self.g_max = g_max

        self.r_theta = self.get_r_theta()
        self.rot_gate = self.get_rot_gate(self.r_theta)  # 获得旋转矩阵

    def get_step(self):
        step = self.C * np.exp(-self.g/self.g_max)
        return step


class DynamicRotGate_F(RotGate):
    """
    基于适应度的动态旋转角策略
    """
    def __init__(self, *, qubit_i, x_i, best_i, f_current, f_best):
        super().__init__(qubit_i=qubit_i, x_i=x_i, best_i=best_i)

        self.theta_min = 0.005 * np.pi
        self.theta_max = 0.01 * np.pi
        self.f_current = f_current
        self.f_best = f_best

        self.r_theta = self.get_r_theta()
        self.rot_gate = self.get_rot_gate(self.r_theta)  # 获得旋转矩阵

    def get_step(self):
        step = self.theta_min+(self.theta_max-self.theta_min)*(abs(self.f_current-self.f_best)/max(abs(self.f_current), abs(self.f_best)))
        return step



if __name__ == '__main__':
    qubit_i, x_i, best_i = (0.71, 0.71), 0, 1
    direction_kargs = dict(qubit_i=qubit_i, x_i=x_i, best_i=best_i)
    x = RotGate.get_direction(**direction_kargs)
    print("方向为：", x)

    # s_r = StaticRotGate(step=0.5*np.pi, qubit_i=qubit_i, x_i=x_i, best_i=best_i)
    # print("theta角为：", str_pi_format(s_r.get_r_theta()))
    # a = np.round(s_r.rot_gate, 2)
    # print(a)


    g_max = 100
    g_list = range(1,100,10)

    list_1 = []
    list_2 = []
    for g in g_list:
        d_r_g = DynamicRotGate_GL(**direction_kargs,
                                  g_max=g_max, g=g)
        d_r_ge = DynamicRotGate_GE(**direction_kargs,
                                   g_max=g_max, g=g)
        list_1.append(d_r_g.r_theta)
        list_2.append(d_r_ge.r_theta)
        print("线性型：方向为{}， 旋转角为{}".format(d_r_g.direction, str_pi_format(d_r_g.r_theta)))
        print("指数型：方向为{}， 旋转角为{}".format(d_r_ge.direction, str_pi_format(d_r_ge.r_theta)))
        # print(np.around(d_r_g.rot_gate, 5))
        # print(np.dot(d_r_g.rot_gate, list(qubit_i)))

    print(np.array(g_list))
    print(np.array(list_1))
    print(np.array(list_2))

    plt.plot(g_list, list_1, 'y', label="linear")
    plt.plot(g_list, list_2, 'r', label="exponential")
    plt.legend()  # 展示图例
    plt.xlabel('Generation')  # 给 x 轴添加标签
    plt.ylabel('Theta')  # 给 y 轴添加标签
    plt.show()


    # f_best = -1
    # # f_list = range(-100, 0, 10)
    # f_list = np.linspace(-100, -1, 20)
    # print(f_list)
    # print('--------------------------------------------')
    # for f_current in f_list:
    #     print("当前适应度为{}，最佳适应度为{}".format(f_current, f_best))
    #     d_r_f = DynamicRotGate_F(qubit_i=qubit_i, x_i=x_i, best_i=best_i,
    #                              f_current=f_current, f_best=f_best)
    #     print("方向为{}， 旋转角为{}".format(d_r_f.direction, str_pi_format(d_r_f.r_theta)))
    #     print(np.around(d_r_f.rot_gate, 5))
    #     print(np.dot(d_r_f.rot_gate, list(qubit_i)))