import numpy as np
from tqdm import tqdm
import copy
import random
import math
import matplotlib.pyplot as plt
from itertools import product


class point:
    def __init__(self, t_site=[], t_F_val=[]):
        self.site = t_site
        self.F_val = t_F_val
        self.n_p = 0  # 能支配这个个体的个体数量
        self.S_p = []  # 这个个体所支配的个体的集合
        self.crowding_distance = float('inf')
        self.rank = -1  # 其所位于的支配集
        self.coding = ''

    def __lt__(self, other):
        if self.rank < other.rank:
            return True
        if self.rank == other.rank and self.crowding_distance > other.crowding_distance:
            return True
        return False


class weight:
    def __init__(self, weight_vector=[], B=[], bh=-1):
        self.weight_vector = weight_vector
        self.B = B
        self.bh = bh


# F为要优化的多个函数的lsit,H为权重划分程度，T为权重向量邻域的权重向量数
def MOEAD(F, x_num, x_l=[], x_r=[], iteration_num=25000, P_size=500, coding_length=20, select_rate=0.6,
          crossover_rate=0.7, variation_rate=0.002, P_true=None, problem_name='', H=12, T=5,real_encoding=True):
    F_num = len(F)  # 目标函数的数量
    weight_vectors = get_weight_vectors(F_num, H)  # 权重向量
    N = len(weight_vectors)  # 权重向量的数量
    ep = []  # 用来保存优秀的个体
    P = _init_pop(x_l=x_l, x_r=x_r, P_size=N)  # 初始化种群
    get_F_val(P, F)
    z = np.ones(len(F)) * float('inf')
    for tpoint in P:
        for i in range(len(z)):
            z[i] = min(tpoint.F_val[i], z[i])
    show_fig(P, P_true)
    weight_vectors_d = np.zeros([N, N])  # 权重向量之间的距离
    for i in range(N):
        for j in range(i + 1, N):
            td = 0
            for k_1, k_2 in zip(weight_vectors[i].weight_vector, weight_vectors[j].weight_vector):
                td += (k_2 - k_1) ** 2
            td = math.sqrt(td)
            weight_vectors_d[i][j] = td
            weight_vectors_d[j][i] = td
    weight_vectors_d = weight_vectors_d.argsort(axis=1)  # 按行排序
    for i, tweight in enumerate(weight_vectors):
        tweight.B = []
        for j in range(1, min(T + 1, len(weight_vectors) - 1)):
            tweight.B.append(weight_vectors[weight_vectors_d[i][j]])
    for epoch in tqdm(range(iteration_num + 15)):
        # 更新
        for tpoint, tweight in zip(P, weight_vectors):
            tP = []
            for tindex in np.random.choice(len(tweight.B), 2, replace=False):
                tP.append(P[tweight.B[tindex].bh])
            Q = GA(tP, x_l, x_r, coding_length, select_rate=1, crossover_rate=1,
                   variation_rate=1 / (len(tP[0].site) ), real_encoding=real_encoding)[np.random.choice(2)]
            get_F_val([Q], F)
            for i in range(len(z)):
                if z[i] > Q.F_val[i]:
                    z[i] = Q.F_val[i]
            for tweight_neighbor in tweight.B:
                tindex = tweight_neighbor.bh
                if g_te(Q, tweight_neighbor, z) < g_te(P[tindex], tweight_neighbor, z):
                    P[tindex] = copy.deepcopy(Q)
            # ep.append(Q)
            non_dominate_add(ep,Q)
        # ep = fast_non_dominated_sort(ep, just_one=True)[0]
        # if epoch % 1 == 0:
        #     qu_chong(ep)
        find_same_point(P)
        if epoch % 50 == 0:
            show_fig(ep, P_true, title=f'{problem_name}_第{epoch}代')
            print(f'{problem_name}_在第{epoch}代：len(ep)={len(ep)},len(P)={len(P)}')
            evaluation(ep, P_true)


# 将tpoint加入到P,保持P为非支配集（没有个体支配其它个体）
def non_dominate_add(P,tpoint):
    i=0
    while i<len(P):
        if a_dominate_b(tpoint,P[i]):
            P.remove(P[i])
            continue
        if a_dominate_b(P[i],tpoint):
            return
        pan=True
        for x_1,x_2 in zip(tpoint.site,P[i].site):
            if not (x_1==x_2):
                pan=False
                break
        if pan:
            return
        i+=1
    P.append(tpoint)


# 判断a是否支配b
def a_dominate_b(a,b):
    m=np.asarray(a.F_val)
    n=np.asarray(b.F_val)
    if (not (False in (m<=n))) and (True in (m<n)):
        return True
    return False


def qu_chong(P):
    for i in range(len(P)):
        j = i + 1
        while j < len(P):
            pan = True
            for x_1, x_2 in zip(P[i].site, P[j].site):
                if not (x_1 == x_2):
                    pan = False
                    break
            if pan:
                P.remove(P[j])
            else:
                j += 1
    return


def g_te(tpoint, tweight, tz):
    t_value = -float('inf')
    for i in range(len(tz)):
        t_value = max(t_value, tweight.weight_vector[i] * abs(tpoint.F_val[i] - tz[i]))
    return t_value
    # r=0
    # for i in range(len(tpoint.F_val)):
    #     r+=tweight.weight_vector[i]*tpoint.F_val[i]
    # return r


# 根据目标函数的数量与H来生成权重向量
def get_weight_vectors(m, H):
    weight_vectors = []
    for i, t in enumerate(np.asarray(get_m_H(m, H)) / H):
        weight_vectors.append(weight(weight_vector=t, bh=i))
    return weight_vectors


def get_m_H(m, H):
    if m == 1:
        return [[H]]
    r = []
    for i in range(H + 1):
        for j in get_m_H(m - 1, H - i):
            r.append([i] + j)
    return r


# 显示图形，这里仅以F[0]为横坐标，F[1]为纵坐标
def show_fig(P, P_true, title=''):
    # print(f'len(P)={len(P)},len(P_true)={len(P_true)}')
    color = []
    F_0 = []
    F_1 = []
    if not (P_true is None):
        data_x = []
        data_y = []
        for tpoint in P_true:
            data_x.append(tpoint.F_val[0])
            data_y.append(tpoint.F_val[1])
        F_0 = [] + data_x
        F_1 = [] + data_y
        color = ['red'] * len(data_x)
    for tpoint in P:
        F_0.append(tpoint.F_val[0])
        F_1.append(tpoint.F_val[1])
    color += ['blue'] * len(P)
    # F_0+=data_ZDT1_x
    # F_1+=data_ZDT1_y
    # color+=['red']*len(data_ZDT1_x)

    plt.scatter(x=F_0, y=F_1, color=color)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.title(title)
    plt.show()


# F为要优化的多个目标函数的list,x_num为变量数,各变量的取值范围为[x_l,x_r],总群大小为P_size
def NSGA2(F, x_num, x_l=[], x_r=[], iteration_num=25000, P_size=500, coding_length=20, select_rate=0.6,
          crossover_rate=0.7, variation_rate=0.002, P_true=None, problem_name='',real_encoding=False):
    # F_num = len(F)  # 要优化的目标函数的数量
    # 初始种群
    P = _init_pop(x_l, x_r, P_size)
    # print(f'P[0].site={P[0].site},P[0].site.len={len(P[0].site)}')
    for i in tqdm(range(iteration_num + 100)):
        # print('!!!!!!!!')
        # 利用遗传算法生成子代种群
        # print(1)
        Q = GA(P, x_l, x_r, coding_length, select_rate, crossover_rate, variation_rate=1/len(P[0].site),real_encoding=real_encoding)
        # print(2)
        R = P + Q  # 原种群与新产生的种群混在一起
        get_F_val(R, F)  # 计算各个个体的在各个函数下的值
        # if i % 10 == 0:
        #     show_fig(R, P_true, title=f'{problem_name}_第{i}代_父子2代种群之和')
        # print(3)
        find_same_point(R)
        F_sum = fast_non_dominated_sort(R)  # 求支配集
        # print(4)
        find_same_point(R)
        P_new = []
        for F_i in F_sum:
            if len(P_new) + len(F_i) > P_size:
                break
            P_new += F_i
        # print(5)
        find_same_point(P_new)
        crowding_distance_assignment(F_i)
        # print(6)
        F_i_sorted = sorted(F_i, key=lambda t: -t.crowding_distance)
        # print(f'len(F_i_sorted)={len(F_i_sorted)}')
        # for tpoint in F_i_sorted:
        #     print(tpoint.crowding_distance)
        P_new += F_i_sorted[:P_size - len(P_new)]
        # print(f'len(P)={len(P)},len(P_new)={len(P_new)}')
        P = copy.deepcopy(P_new)
        # print(7)
        find_same_point(P_new)
        # plt.ion()
        if i % 50 == 0:
            show_fig(P_new, P_true, title=f'{problem_name}_第{i}代')
            print(f'{problem_name}_在第{i}代：')
            evaluation(P_new, P_true)


# 初始化种群
def _init_pop(x_l=[], x_r=[], P_size=100):
    P = []
    for i in range(P_size):
        t_site = np.random.random(len(x_l))
        for j in range(len(x_l)):
            t_site[j] = t_site[j] * (x_r[j] - x_l[j]) + x_l[j]
        P.append(point(t_site=t_site))
    return P


def get_F_val(P, F):
    for tpoint in P:
        F_val = []
        for i in range(len(F)):
            # print(tpoint.site)
            F_val.append(F[i](tpoint.site))
        tpoint.F_val = F_val


# 检测重复
def find_same_point(P):
    return  # 下面的是测试时用的，现在直接跳过
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            if P[i] == P[j]:
                print(f'P[{i}]==P[{j}]')


# 求支配集
def fast_non_dominated_sort(tP, just_one=False):
    P = copy.deepcopy(tP)
    # 检测重复
    find_same_point(P)
    # 初始化
    for tpoint in P:
        tpoint.S_p = []
        tpoint.n_p = 0
    # 计算各个个体的n_p并将相应的个体加入到S_p
    for i in range(len(P)):
        for j in range(len(P)):
            a = np.asarray(P[i].F_val) <= np.asarray(P[j].F_val)
            b = np.asarray(P[i].F_val) < np.asarray(P[j].F_val)
            # print(f'P[i].F_val={P[i].F_val},P[j].F_val={P[j].F_val},a={a}')
            if (not (False in a)) and (True in b):  # a全为True的话
                P[i].S_p.append(P[j])
                P[j].n_p += 1
    S = []
    count = 0  # 用来记录当前分配到哪个支配集了
    while len(P) > 0:
        tS = []
        for t_point in P:
            if t_point.n_p == 0:
                tS.append(t_point)
                t_point.rank = count
                # P.remove(t_point)
        for t_point in tS:
            for tt_point in t_point.S_p:
                tt_point.n_p -= 1
            P.remove(t_point)
        S.append(tS)
        if just_one:
            break
        count += 1
    return S


# 求拥挤度
def crowding_distance_assignment(I):
    if len(I) == 0:
        return
    for tI in I:
        tI.crowding_distance = 0
    for i in range(len(I[0].F_val)):
        I_sorted = sorted(I, key=lambda tI: tI.F_val[i])
        I_sorted[0].crowding_distance = float('inf')
        I_sorted[-1].crowding_distance = float('inf')
        fm = I_sorted[-1].F_val[i] - I_sorted[0].F_val[i]
        for j in range(1, len(I_sorted) - 1):
            I_sorted[j].crowding_distance += (I_sorted[j + 1].F_val[i] - I_sorted[j - 1].F_val[i]) / fm


# 用遗传算法(仅选择、交叉、变异)生成新的种群,这里的交叉与突变率是针对选择出来的子种群而言的,返回生成的新种群
def GA(P, x_l, x_r, coding_length, select_rate=0.6, crossover_rate=0.7, variation_rate=0.001, real_encoding=False):
    # 产生要操作的子种群
    P_child = _GA_select(P, select_rate)
    # 对子种群中的各个个体进行编码
    _GA_encode(P_child, x_l, x_r, coding_length, real_encoding)
    # 交叉
    _GA_crossover(P_child, crossover_rate, coding_length, real_encoding, x_l, x_r)
    # 变异
    _GA_variation(P_child, variation_rate, real_encoding, x_l, x_r)
    # 解码
    _GA_decode(P_child, x_l, x_r, coding_length, real_encoding)
    return P_child


# 编码
def _GA_encode(P, x_l, x_r, coding_length=20, real_encoding=False):
    if real_encoding:
        return
    for tpoint in P:
        tpoint.coding = ''
        for i, tsite in enumerate(tpoint.site):
            tpoint.coding += _dec_to_bin(tsite, x_l[i], x_r[i], coding_length)[2:].zfill(coding_length)
        # print(f'len(tpoint_coding)={len(tpoint.coding)}')


# 将一个数进行二进制编码
def _dec_to_bin(x, x_l, x_r, coding_length=20):
    tx = (x - x_l) / (x_r - x_l)
    tx *= 2 ** coding_length - 1
    # print(tx)
    tx = int(round(tx))
    # print(tx)
    x_bin = bin(tx)
    return x_bin


# 解码
def _GA_decode(P, x_l, x_r, coding_length=20, real_encoding=False):
    if real_encoding:
        return
    # print(f'len(P)={len(P)},len(x_l)={len(x_l)}')
    for j, tpoint in enumerate(P):
        coding_number = int(len(tpoint.coding) / coding_length)
        tpoint.site = np.zeros([coding_number])
        for i in range(coding_number):
            tpoint.site[i] = _bin_to_dec(tpoint.coding[i * coding_length:(i + 1) * coding_length], x_l[i], x_r[i],
                                         coding_length)


# 将一个二进制数还原成小数
def _bin_to_dec(x_bin, x_l, x_r, coding_length=20):
    tx = int(x_bin, base=2)
    tx /= 2 ** coding_length - 1
    x = tx * (x_r - x_l) + x_l
    return x


# 选择
def _GA_select(P, select_rate):
    P_size = len(P)
    P_child_size = int(len(P) * select_rate)
    P_child_index = np.random.choice(P_size, P_child_size, replace=False)
    P_child = []
    for t_index in P_child_index:
        P_child.append(copy.deepcopy(P[t_index]))
    return P_child


# 交叉,这里是不断选2个进行交叉，直到达到需要的交叉率（不是直接选择）,
def _GA_crossover(P, crossover_rate, coding_length=20, real_encoding=False, x_l=[], x_r=[]):
    for _ in range(math.ceil(len(P) / 2)):
        if random.random() > crossover_rate:
            continue
        point1, point2 = random.sample(P, 2)  # 要进行交叉的2个个体
        if real_encoding:
            eta = 1
            for i in range(len(point1.site)):
                u = random.random()
                if u <= 0.5:
                    y = (2 * u) ** (1 / (1 + eta))
                else:
                    y = (1 / (2 * (1 - u))) ** (1 / (1 + eta))
                tx_1 = 0.5 * ((1 + y) * point1.site[i] + (1 - y) * point2.site[i])
                tx_2 = 0.5 * ((1 - y) * point1.site[i] + (1 + y) * point2.site[i])
                point1.site[i] = np.clip(tx_1, x_l[i], x_r[i])
                point2.site[i] = np.clip(tx_2, x_l[i], x_r[i])
        else:
            rst_index = np.random.choice(len(P[0].site) - 2) + 1  # 要进行交换的染色体的位置
            # print(f'before:len(point1.coding)={len(point1.coding)},len(point2.coding)={len(point2.coding)}')
            # #这个是仅交换一个
            # txx=point1.coding[rst_index*coding_length:(rst_index+1)*coding_length]
            # point1.coding=point1.coding[:rst_index*coding_length]+point2.coding[rst_index*coding_length:(rst_index+1)*coding_length]+point1.coding[(rst_index+1)*coding_length:]
            # point2.coding=point2.coding[:rst_index*coding_length]+txx+point2.coding[(rst_index+1)*coding_length:]
            # 这个是以这个位置为交叉点整个进行交换(单点交叉)
            txx = point1.coding[rst_index * coding_length:]
            point1.coding = point1.coding[:rst_index * coding_length] + point2.coding[rst_index * coding_length:]
            point2.coding = point2.coding[:rst_index * coding_length] + txx
            # print(f'after:len(point1.coding)={len(point1.coding)},len(point2.coding)={len(point2.coding)}')


# 变异,这里的变异率是针对基因数的，不是个体数
def _GA_variation(P, variation_rate, real_encoding=False, x_l=[], x_r=[]):
    # P_size = len(P)  # 种群大小
    # gene_num = len(P[0].coding)  # 单个个体的基因数
    # variation_size = int(P_size * gene_num * variation_rate)  # 总共需要进行变异操作的基因数
    # count = 0  # 用来记录总共产生了多少次变异
    for _ in range(len(P) * len(P[0].site) if real_encoding else len(P) * len(P[0].coding)):
        tpoint = random.sample(P, 1)[0]
        if real_encoding:
            eta_m = 20
            for i in range(len(tpoint.site)):
                if random.random() > variation_rate:
                    continue
                delta1 = (tpoint.site[i] - x_l[i]) / (x_r[i] - x_l[i])
                delta2 = (x_r[i] - tpoint.site[i]) / (x_r[i] - x_l[i])
                u = random.random()
                if u <= 0.5:
                    delta = (2 * u + (1 - 2 * u) * (1 - delta1) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - delta2) ** (eta_m + 1)) ** (1 / (eta_m + 1))
                tpoint.site[i] += np.clip(delta * (x_r[i] - x_l[i]), x_l[i], x_r[i])
            return
        else:
            if random.random() > variation_rate:
                continue
            tcoding_index = random.randint(0, len(P[0].coding) - 1)
            t_str = '0'
            if tpoint.coding[tcoding_index] == '0':
                t_str = '1'
            tpoint.coding = _str_replace_char(tpoint.coding, tcoding_index, t_str)


# 对字符中的指定位置的字符进行替换
def _str_replace_char(str1, wz, str2):
    str_fin = str1[:wz] + str2 + str1[wz + 1:]
    return str_fin


class problem:
    def __init__(self, F=[], x_num=0, x_l=[], x_r=[], P_true=[], problem_name=''):
        self.F = F
        self.x_num = x_num
        self.x_l = x_l
        self.x_r = x_r
        self.P_true = P_true  # 真实的PF上的个体的集合
        self.problm_name = problem_name

    def solve(self, iteration_num=300, P_size=100, coding_length=10, select_rate=1, crossover_rate=1,
              variation_rate=0.002):
        # NSGA2(self.F, self.x_num, self.x_l, self.x_r, iteration_num, P_size, coding_length, select_rate, crossover_rate,
        #       variation_rate, self.P_true, self.problm_name)
        MOEAD(self.F, self.x_num, self.x_l, self.x_r, iteration_num, P_size, coding_length, select_rate, crossover_rate,
              variation_rate, self.P_true, self.problm_name, H=149, T=20)


def get_problem(problem_name=''):
    def g_1(x):
        r = 0
        for tx in x[1:]:
            r += tx
        r *= 9
        r /= (len(x) - 1)
        r += 1
        return r

    def f_1(x):
        return x[0]

    def f_2(x):
        return g_1(x) * (1 - math.sqrt(x[0] / g_1(x)))

    def f_3(x):
        return g_1(x) * (1 - (x[0] / g_1(x)) ** 2)

    def f_4(x):
        return g_1(x) * (1 - math.sqrt(x[0] / g_1(x)) - x[0] / g_1(x) * math.sin(10 * math.pi * x[0]))

    def g_2(x):
        r = 0
        for tx in x[1:]:
            r += tx ** 2 - 10 * math.cos(4 * math.pi * tx)
        return 1 + 10 * (len(x) - 1) + r

    def f_5(x):
        return g_2(x) * (1 - math.sqrt(x[0] / g_2(x)))

    def g_3(x):
        r = 0
        for tx in x[1:]:
            r += tx
        r /= (len(x) - 1)
        r = r ** 0.25
        r *= 9
        r += 1
        return r

    def f_6(x):
        return 1 - math.exp(-4 * x[0]) * (math.sin(6 * math.pi * x[0])) ** 6

    def f_7(x):
        g_value = g_3(x)
        return g_value * (1 - (f_6(x) / g_value) ** 2)

    def get_P_true(file_name):
        data = open(file_name).read()
        data = data.split('\n')
        data_x = []
        data_y = []
        for tdata in data:
            data_x.append(float(tdata.split()[0]))
            data_y.append(float(tdata.split()[1]))
        P_true = []
        for tx, ty in zip(data_x, data_y):
            P_true.append(point(t_F_val=[tx, ty]))
        return P_true

    if problem_name == 'ZDT1':
        x_num = 30
        x_l = [0] * x_num
        x_r = [1] * x_num
        F = [f_1, f_2]
        return problem(F=F, x_num=x_num, x_l=x_l, x_r=x_r, P_true=get_P_true(problem_name + '.txt'),
                       problem_name=problem_name)
    elif problem_name == 'ZDT2':
        x_num = 30
        x_l = [0] * x_num
        x_r = [1] * x_num
        F = [f_1, f_3]
        return problem(F=F, x_num=x_num, x_l=x_l, x_r=x_r, P_true=get_P_true(problem_name + '.txt'),
                       problem_name=problem_name)
    elif problem_name == 'ZDT3':
        x_num = 30
        x_l = [0] * x_num
        x_r = [1] * x_num
        F = [f_1, f_4]
        return problem(F=F, x_num=x_num, x_l=x_l, x_r=x_r, P_true=get_P_true(problem_name + '.txt'),
                       problem_name=problem_name)
    elif problem_name == 'ZDT4':
        x_num = 10
        x_l = [-5] * x_num
        x_r = [5] * x_num
        x_l[0] = 0
        x_r[0] = 1
        F = [f_1, f_5]
        return problem(F=F, x_num=x_num, x_l=x_l, x_r=x_r, P_true=get_P_true(problem_name + '.txt'),
                       problem_name=problem_name)
    elif problem_name == 'ZDT6':
        x_num = 10
        x_l = [0] * x_num
        x_r = [1] * x_num
        F = [f_6, f_7]
        return problem(F=F, x_num=x_num, x_l=x_l, x_r=x_r, P_true=get_P_true(problem_name + '.txt'),
                       problem_name=problem_name)


# 计算两个point之间的距离
def get_distance(point_1, point_2):
    r = 0
    for x_1, x_2 in zip(point_1.F_val, point_2.F_val):
        r += (x_1 - x_2) ** 2
    return math.sqrt(r)


def evaluation(P, P_true=[]):
    P_distance = np.zeros([len(P), len(P)])  # 用来记录P中各个个体之间的距离
    P_P_true_distance = np.zeros([len(P), len(P_true)])  # 用来记录P与P_true中各个个体之间的距离

    def get_P_distance(P):
        # P_distance=np.zeros([len(P),len(P)])
        for i in range(len(P)):
            P_distance[i][i] = float('inf')
            for j in range(i + 1, len(P)):
                t_distance = get_distance(P[i], P[j])
                # if t_distance==0:
                #     print('t_distance==0!!!!!!')
                P_distance[i][j] = t_distance
                P_distance[j][i] = t_distance

    def get_P_P_true_distance(P, P_true):
        # P_P_true_distance=np.zeros([len(P),len(P_true)])
        for i in range(len(P)):
            for j in range(len(P_true)):
                t_distance = get_distance(P[i], P_true[j])
                P_P_true_distance[i][j] = t_distance

    # Spacing:度量每个解到其他解的最小距离的标准差。Spacing值越小，说明解集越均匀
    def Spacing(P):
        get_P_distance(P)
        P_distance_np = np.asarray(P_distance)
        P_distance_np.sort(axis=1)  # 按行进行排序
        d = P_distance_np.take(0, axis=1)
        return d.std(ddof=1)  # 求无偏标准样本差

    # GD:解集P中的每个点到参考集P *中的平均最小距离表示。GD值越小，表示收敛性越好
    def GD(P, P_true):
        get_P_P_true_distance(P, P_true)
        P_P_true_distance_np = np.asarray(P_P_true_distance)
        P_P_true_distance_np.sort(axis=1)  # 按行进行排序
        # print(f'P_P_true_distance={P_P_true_distance}')
        d = P_P_true_distance.take(0, axis=1)
        # print(f'd={d}')
        return math.sqrt((d ** 2).sum()) / len(P)

    # 反转世代距离（IGD，Inverted Generational Distance）：每个参考点到最近的解的距离的平均值。IGD值越小，说明算法综合性能越好
    def IGD(P, P_true):
        get_P_P_true_distance(P, P_true)
        P_P_true_distance_np = np.asarray(P_P_true_distance)
        P_P_true_distance_np.sort(axis=0)  # 按列进行排序
        d = P_P_true_distance.take(0, axis=0)  # 取第一行
        return d.sum() / len(P)

    # print(f'len(P)={len(P)},len(P_true)={len(P_true)}')
    sp_value = Spacing(P)
    gd_value = GD(P, P_true)
    igd_value = IGD(P, P_true)
    print(f'sp={sp_value},gd={gd_value},igd={igd_value}')


if __name__ == '__main__':
    # problem_name='ZDT3'
    # ZDT1=get_problem(problem_name)
    # file_ZDT1=problem_name+'.txt'
    # data_ZDT1=open(file_ZDT1).read()
    # data_ZDT1=data_ZDT1.split('\n')
    # data_ZDT1_x=[]
    # data_ZDT1_y=[]
    # for tdata in data_ZDT1:
    #     data_ZDT1_x.append(float(tdata.split()[0]))
    #     data_ZDT1_y.append(float(tdata.split()[1]))
    # ZDT1.solve()
    #
    problem_names = ['1', '2', '3', '4', '6']
    problem_names = ['ZDT' + ts for ts in problem_names]
    for problem_name in problem_names[1:]:
        print(f'problem_name={problem_name}')
        my_problem = get_problem(problem_name)
        my_problem.solve()
