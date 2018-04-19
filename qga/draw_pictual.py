# -*- coding: utf-8 -*-
"""
Created on 2018/4/18 21:57

@author: dream01
"""
import matplotlib.pyplot as plt
import numpy as np


def get_pictual():
    """
    读取csv文件
    :return:
    """
    data_txt = np.loadtxt("mydata2.csv", dtype=np.str, delimiter=",")
    data = data_txt[1:, :-1].astype(np.float)
    theta = data[:, 0]

    avg = data[:, 1]
    median = data[:, 2]
    best = data[:, 3]
    global_best = data[:, 4]
    better_count = data[:, 5]
    first_better_generation = data[:, 6]
    best_in_many_times = data[:, 7]

    plt.subplot(3, 2, 1)
    plt.plot(theta, avg, label="avg")
    plt.legend()  # 展示图例
    plt.xlabel('theta')  # 给 x 轴添加标签
    plt.ylabel('fitness')
    #
    plt.subplot(3, 2, 2)
    plt.plot(theta, median, label="median")
    plt.legend()  # 展示图例
    plt.xlabel('theta(pi)')
    plt.ylabel('fitness')

    plt.subplot(3, 2, 3)
    plt.plot(theta, best, label="best")
    plt.legend()  # 展示图例
    plt.xlabel('theta(pi)')
    plt.ylabel('fitness')

    plt.subplot(3, 2, 4)
    plt.plot(theta, better_count, label="better_count")
    plt.legend()  # 展示图例
    plt.xlabel('theta(pi)')
    plt.ylabel('count')
    #
    plt.subplot(3, 2, 5)
    plt.plot(theta, first_better_generation, label="first_better_generation")
    plt.legend()  # 展示图例
    plt.xlabel('theta(pi)')
    plt.ylabel('generation')

    plt.subplot(3, 2, 6)
    plt.plot(theta, best_in_many_times, label="best_in_many_times")
    plt.legend()  # 展示图例
    plt.xlabel('theta(pi)')
    plt.ylabel('fitness')

    plt.show()


if __name__ == '__main__':
    get_pictual()