# -*- coding: utf-8 -*-
"""
Created on 2018/4/9 3:40

@author: dream01
"""
import numpy as np
import matplotlib.pyplot as plt

def to_decimal(a_list):
    l = len(a_list)
    x = 0
    for i, bit in enumerate(a_list):
        x += (bit * pow(2, l - i - 1))
    return x

def common_decoder(chrom):
    """
    范围决定于染色体长度
    常用的双变量解码函数，将染色体对半分，前一部分表示变量x,后一部分表示变量y
    对传入的二进制串（染色体）进行解码，总共m位,均分为m1,m2两个位，各个部分的第一位代表符号，0 为负 1为正，接下来的位则表示数
    如：m = 24, 总共两个变量, x: n=12, y: n =12, x的第1位是符号为,剩下11位表示 2^11 =2048个数
    2^11=2048
    """
    length = len(chrom)         # 16
    v_length = length // 2      # 8
    x_genom = chrom[1:v_length]
    y_genom = chrom[v_length + 1:]
    x_sign = 1 if chrom[0] else -1
    y_sign = 1 if chrom[v_length] else -1
    x = to_decimal(x_genom)
    y = to_decimal(y_genom)
    x = x_sign * x
    y = y_sign * y
    return x, y


def str_pi_format(theta):
    """

    :param theta:
    :return:
    """
    return str(theta/np.pi)+'pi'


def draw_picual(x, y):
    plt.figure()
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    # a = np.array([1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0])
    # j = [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
    # a = np.array(j)
    # x, y = common_decoder(a)
    # print(x/1000, y/1000)
    print(to_decimal([1,0,0,0]))