# -*- coding: utf-8 -*-
"""
Created on 2018/4/15 0:55

@author: dream01
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from utils import to_decimal, common_decoder


class Question(object):
    def func(self, args):
        pass

    def decoder(self, binary_strings):
        pass

    def get_fargs_and_result(self, args):
        decoded_args = self.decoder(args)       # 将二进制串解码
        func_result = self.func(decoded_args)   # 将解码的二进制串带入对应问题
        # **print("解码信息", args, decoded_args, func_result)
        return decoded_args, func_result

    def draw(self):
        pass

    def draw_2d(self, x):
        y = self.func((x,))
        plt.figure()
        plt.plot(x, y)
        y_max = y.max()
        index = np.where(y == y_max)[0][0]
        print("y最大值：{}， 最大值的index：{}, x={}, y_max={}".format(y_max, index, x[index], y[index]))
        plt.show()

    def draw_3d(self, args):
        x, y = args

        fig = plt.figure()
        ax = Axes3D(fig)
        X, Y = np.meshgrid(x, y)
        Z = self.func((X, Y))
        Z_min, Z_max = Z.min(), Z.max()
        min_y_index, min_x_index = np.where(Z == Z_min)
        max_y_index, max_x_index = np.where(Z == Z_max)
        print("区间： x 属于[{},{}]， y属于[{},{}]".format(x[0], x[-1], y[0], y[-1]))
        print("最小值为{}，(x, y) = ({}, {})".format(Z_min, x[min_x_index[0]], y[min_y_index[0]]))
        print("最大值为{}，(x, y) = ({}, {})".format(Z_max, x[max_x_index[0]], y[max_y_index[0]]))
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
        ax.contourf(X, Y, Z, zdir='x', offset=-2.3, cmap='rainbow')  # 画热力图
        # ax.set_zlim(-1,1)
        ax.set_xlim(-2.3, 2.3)
        # ax.set_ylim(-12, 12)
        plt.show()


class Q1(Question):
    """
    f(x)=abs((x-5)/(2+sin(x))) x 在区间[0,15], x取11的时候f(x)取最大值
    m = 4， 编码方式为4
    """
    m = 4
    better = 590                            # 定义的较优解

    def func(self, args):
        x, = args
        return abs((x-5)/(2+np.sin(x)))*100
        # return x+100

    def decoder(self, binary_strings):
        """
        解码器，将二进制串解码，如果是单变量，这返回一个数，双变量则有两个
        """
        return to_decimal(binary_strings),

    def draw(self):
        x = np.linspace(0, 15, 100)
        super().draw_2d(x)


class Q1_1(Question):
    """
    f(x)=abs((x-5)/(2+sin(x))) x 在区间[0,20.47]
    x=17.36左右的时候f(x)取最大值12.31
    m = 11， 编码长度为11, 2^11 = 2048
    """
    m = 11

    def func(self, args):
        x, = args
        return abs((x-5)/(2+np.sin(x)))*1000
        # return x+100

    def decoder(self, binary_strings):
        """
        解码器，将二进制串解码，如果是单变量，这返回一个数，双变量则有两个
        [0,2047] => [0,20.48]
        """
        return to_decimal(binary_strings)/100,


class DeJong(Question):
    """
    DeJong 函数： 双变量x,y， 均属于区间[-2.048, 2.048]， 有极小值f(1,1)=0，
    m = 24, 编码长度24位,每个变量12位，前12位代表x，后12位代表y
    12位编码中，第1位代表符号：0 为负 1为正，接下来的11位则代表数，2^11=2048
    """
    m = 24
    better = -0.01                                    # 定义的更优解
    good = -0.1                                       # 定义的优解

    def func(self, args):
        x, y = args
        return -(100 * np.square(np.square(x) - y) + np.square(1 - x))

    def decoder(self, binary_strings):
        d_x, d_y = common_decoder(binary_strings)       # 现在d_x, d_y在区间[-2048,2048]中
        return d_x/1000, d_y/1000                       # 要转变为区间[-2.048,2.048]

    def draw(self):
        x = np.linspace(-2.047, 2.047, 100)
        y = np.linspace(-2.047, 2.047, 100)
        super().draw_3d((x,y))


class GoldStein(Question):
    """
    GoldStein 函数：双变量x,y, 均都处于[-2.048,2.048]的区间, f(0,-1)极小值3
    """
    m = 24

    def func(self, args):
        x, y = args
        t1 = 1 + np.square((x + y + 1)) * (19 - 14 * x + 3 * np.square(x) - 14 * y + 6 * x * y + 3 * np.square(y))
        t2 = 30 + np.square(2 * x - 3 * y) * (18 - 32 * x + 12 * np.square(x) + 48 * y - 36 * x * y + 27 * np.square(y))
        return -(t1 * t2)

    def decoder(self, binary_strings):
        d_x, d_y = common_decoder(binary_strings)       # 现在d_x, d_y在区间[-2047,2047]中
        return d_x/1000, d_y/1000                       # 要转变为区间[-2.047,2.047]

    def draw(self):
        x = np.arange(-2.047, 2.047, 0.01)
        y = np.arange(-2.047, 2.047, 0.01)
        super().draw_3d((x,y))

if __name__ == '__main__':
    q1 = Q1()
    r1 = q1.get_fargs_and_result([0,0,1,1])
    print(q1.m, r1)
    q1.draw()
    # dj = DeJong()
    # dj.draw()

    # gs = GoldStein()
    # gs.draw()
    # a = gs.func((0.002999999999956149, -0.9970000000000225))
    # print(a)