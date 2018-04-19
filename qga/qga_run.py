# -*- coding: utf-8 -*-
"""
Created on 2018/4/18 16:21

@author: dream01
"""

from qga.q_pop import QuantumPopulation
from qga.question import Q1, Q1_1, DeJong, GoldStein
import numpy as np
from utils import str_pi_format

from concurrent.futures import ProcessPoolExecutor as Pool


def run_QGA(s_step=0.025*np.pi):
    q_pop = QuantumPopulation(question=DeJong(), step=s_step)
    q_pop.init_population()             # 初始化种群
    q_pop.measure()                     # 初始观测
    q_pop.evaluate_population()         # 初始的适应度评估
    q_pop.show_population_infor()
    while q_pop.generation < q_pop.max_generation:
        q_pop.generation += 1
        print('-----------------------------------第{}代， 幅值{}------------------------------------------'.format(q_pop.generation, str_pi_format(s_step)))
        q_pop.update_quantum_population()       # 更新量子种群
        q_pop.measure()                         # 量子观测
        q_pop.evaluate_population()             # 评价该量子种群
        q_pop.show_population_infor()
    print('')
    print("最优适应度为：{}, 编码后的值为{}, 染色体表示为{}".format(q_pop.best_fitness, q_pop.best_solution,
                                                 q_pop.best_chrom))
    print('最先取得较优解的代数为', q_pop.first_better_fitness_generation)
    # q_pop.draw_pictual_2()
    print('**************************************************************************')
    return q_pop


def qga_debug():
    q1 = QuantumPopulation(Q1())
    q1.init_population()
    q1.measure()
    q1.evaluate_population()  # 初始的适应度评估
    q1.show_population_infor()


def run_many_times(step):
    times = 20
    avg_fitness_list = np.empty(times)                 # 进化到最后一代的平均适应度
    median_fitness_list = np.empty(times)              # 进化到最后一代的适应度中位数
    best_fitness_list = np.empty(times)                # 进化到最后一代的最优适应度
    global_best_fitness_list = np.empty(times)         # 进化到最后一代的全局最优解

    better_fitness_count_list = np.empty(times)         # 进化到最后一代的获得较优解次数
    first_better_fitness_generation_list = []           # 全局最先获得较优解的代数(有可能为None)

    best_fitness_in_many_times = None                   # 30代里面最好的解
    best_fitness_solution_in_many_times = None          # 30代里面最好解的xy解码值
    best_fitness_chrom_in_many_times = None             # 30代里面最好解的染色体编码

    for i in range(times):
        q_pop = run_QGA(s_step=step)

        avg_fitness_list[i] = q_pop.avg_fitness_of_each_generation[-1]
        median_fitness_list[i] = q_pop.median_fitness_of_each_generation[-1]
        best_fitness_list[i] = q_pop.best_fitness_of_each_generation[-1]

        global_best_fitness_list[i] = q_pop.best_fitness

        better_fitness_count_list[i] = q_pop.better_fitness_count_of_each_generation[-1]
        if q_pop.first_better_fitness_generation:
            first_better_fitness_generation_list.append(q_pop.first_better_fitness_generation)
        if not best_fitness_in_many_times or q_pop.best_fitness > best_fitness_in_many_times:
            best_fitness_in_many_times = q_pop.best_fitness
            best_fitness_solution_in_many_times = q_pop.best_solution
            best_fitness_chrom_in_many_times = str(q_pop.best_chrom)

    return str_pi_format(step), {
        "avg_fitness" : np.average(avg_fitness_list),
        "median_fitness" : np.average(median_fitness_list),
        "best_fitness": np.average(best_fitness_list),
        "global_best_fitness": np.average(global_best_fitness_list),

        "better_fitness_count" : np.average(better_fitness_count_list),
        "first_better_fitness_generation" : np.average(first_better_fitness_generation_list) if first_better_fitness_generation_list else -1,       # ************************

        "best_fitness_in_many_times" : best_fitness_in_many_times,
        "best_fitness_solution_in_many_times" : best_fitness_solution_in_many_times,
        "best_fitness_chrom_in_many_times" : best_fitness_chrom_in_many_times,
    }


def my_test_1():
    # step_list = np.pi * np.linspace(0.001, 0.5, 50)
    step_list = np.pi * np.linspace(0.001, 0.05, 30)

    pool = Pool(max_workers=3)
    infor_list = pool.map(run_many_times, step_list)     # 使用多进程改进

    # infor_list = [run_many_times(step) for step in step_list]

    infor_list = sorted(list(infor_list))
    print(infor_list)

    with open('mydata2.csv', 'w') as f:
        f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(
            'theta', 'avg', 'median', 'best', 'global_best',
            'better_count', 'first_better_generation',
            'best_in_many_times', 'best_x_in_many_times', 'best_y_in_many_times',
            'best_chrom_in_many_times'
        ))

    for s, d in infor_list:
        for k,v in d.items():
            print(k, v)
        with open('mydata2.csv', 'a') as f:
            f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(
                s.replace('pi', ''), d['avg_fitness'], d['median_fitness'], d['best_fitness'], d['global_best_fitness'],
                d['better_fitness_count'], d['first_better_fitness_generation'],
                d['best_fitness_in_many_times'], d['best_fitness_solution_in_many_times'][0], d['best_fitness_solution_in_many_times'][1],
                d['best_fitness_chrom_in_many_times']
            ))


# def x():
#     theta_avg_fitness = []
#     theta_median_fitness = []
#     theta_best_fitness = []
#     theta_global_best_fitness = []
#
#     theta_better_fitness_count = []
#     theta_first_better_fitness_generation = []
#
#     theta_best_fitness_in_many_times = []
#     theta_best_fitness_solution_in_many_times = []
#     theta_best_fitness_chrom_in_many_times = []
#
#     for s,d in infor_list:
#         # for k,v in d.items():
#         theta_avg_fitness.append(d['avg_fitness'])
#         theta_median_fitness.append(d['median_fitness'])
#         theta_best_fitness.append(d['best_fitness'])
#         theta_global_best_fitness.append(d['global_best_fitness'])
#
#         theta_better_fitness_count.append(d['better_fitness_count'])
#         theta_first_better_fitness_generation.append(d['first_better_fitness_generation'])
#
#         theta_best_fitness_in_many_times.append(d['best_fitness_in_many_times'])
#         theta_best_fitness_solution_in_many_times.append(d['best_fitness_solution_in_many_times'])
#         theta_best_fitness_chrom_in_many_times.append(d['best_fitness_chrom_in_many_times'])

if __name__ == '__main__':
    # run_QGA(s_step=0.35 * np.pi)
    # run_QGA()
    c = True
    while c:
        a = run_QGA()
        if a.first_better_fitness_generation:
            c = False
            a.draw_pictual_2()
    # theta, d = run_many_times(step=0.025*np.pi)
    # for k, v in d.items():
    #     print(k, v)

    # my_test_1()