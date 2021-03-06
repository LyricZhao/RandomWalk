import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import os
import random

from functools import partial
from scipy import optimize, stats


# For fast debugging
global_steps_limit = 4000
global_simulate_times = 10000
global_brownian_delta_t = 0.00005


def choice(p: float) -> bool:
    assert 0 <= p <= 1
    return random.uniform(0, 1) <= p


def shift_choice(p: float) -> int:
    return 1 if choice(p) else -1


def simulate(a1: int, a2: int, p1: float, p2: float,
             steps_limit: int = global_steps_limit,
             return_none: bool = False) -> int:
    if a1 == a2:
        return 0
    assert (a2 - a1) % 2 == 0, 'The distance between two points must be even'
    steps = 0
    for i in range(steps_limit):
        steps = steps + 1
        a1 = a1 + shift_choice(p1)
        a2 = a2 + shift_choice(p2)
        if a1 == a2:
            return steps
    return None if return_none else steps


def simulate_brownian(a1: float, a2: float,
                      steps_limits: int = global_steps_limit,
                      delta_t: float = global_brownian_delta_t,
                      return_pos: bool = False,
                      return_min: bool = False,
                      return_max: bool = False) -> float:
    if a1 == a2:
        if return_pos or return_min or return_max:
            return a1
        return 0
    steps = 0
    sqrt_delta_t = math.sqrt(delta_t)
    min_p, max_p = min(a1, a2), max(a1, a2)
    for i in range(steps_limits):
        steps = steps + 1
        a1 = a1 + sqrt_delta_t * random.gauss(0, 1)
        a2 = a2 + sqrt_delta_t * random.gauss(0, 1)
        min_p = min(min_p, min(a1, a2))
        max_p = max(max_p, max(a1, a2))
        if abs(a1 - a2) < sqrt_delta_t:
            if return_pos:
                return (a1 + a2) / 2
            if return_min:
                return min_p
            if return_max:
                return max_p
            return steps * delta_t
    return None


def simulate_hit_brownian(a: float,
                          steps_limits: int = global_steps_limit,
                          delta_t: float = global_brownian_delta_t) -> float:
    steps = 0
    sqrt_delta_t = math.sqrt(delta_t)
    for i in range(steps_limits):
        steps = steps + 1
        a = a + sqrt_delta_t * np.random.randn()
        if abs(a) < sqrt_delta_t / 2:
            return steps * delta_t
    return None


def shift_choice_2d() -> [int, int]:
    if choice(0.5):
        return (1, 0) if choice(0.5) else (-1, 0)
    else:
        return (0, 1) if choice(0.5) else (0, -1)


def simulate_2d(a1: (int, int), a2: (int, int), steps_limit: int = global_steps_limit) -> int:
    if a1 == a2:
        return 0
    steps = 0
    for i in range(steps_limit):
        steps = steps + 1
        s1 = shift_choice_2d()
        a1 = (a1[0] + s1[0], a1[1] + s1[1])
        s2 = shift_choice_2d()
        a2 = (a2[0] + s2[0], a2[1] + s2[1])
        if a1 == a2:
            return steps
    return None


def multiple_simulate(bind_func_obj: partial, accumulate: bool = False,
                      simulation_times: int = global_simulate_times,
                      count_success: bool = False) -> ([int], [(int, float)]):
    a, d = [], {}
    assert simulation_times > 0
    success_times = 0
    for i in range(simulation_times):
        x = bind_func_obj()
        if x is not None:
            a.append(x)
            success_times = success_times + 1
            d[x] = d[x] + 1 if x in d else 1
    d = sorted(d.items())
    div = success_times if count_success else simulation_times
    d = list(map(lambda x: (x[0], x[1] / div), d))
    if accumulate:
        for i in range(1, len(d)):
            d[i] = (d[i][0], d[i - 1][1] + d[i][1])
    return a, d


def expectation(bind_func_obj: partial, simulation_times: int = global_simulate_times) -> float:
    a, _ = multiple_simulate(bind_func_obj, False, simulation_times)
    return np.mean(a)


def variance(bind_func_obj: partial, simulation_times: int = global_simulate_times) -> float:
    a, _ = multiple_simulate(bind_func_obj, False, simulation_times)
    return np.var(a, ddof=1)  # should be without bias


def prepare_comb(n: int) -> np.array:
    c = np.zeros((n + 1, n + 1), dtype=float)
    for i in range(n + 1):
        c[i, 0] = 1
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            c[i, j] = c[i - 1, j - 1] + c[i - 1, j]
    return c


def draw(func, x_label: str, y_label: str, ref_func=None, desc: str = '', filename: str = None):
    x, y = [], []
    for a, b in func():
        x.append(a), y.append(b)
    fig = plt.figure()
    plt.title(f'{y_label} towards {x_label} {desc}')
    plt.xlabel(x_label), plt.ylabel(y_label)
    [handles, ], labels = [plt.plot(x, y)], ['Simulation']
    if ref_func is not None:
        handle, = plt.plot(x, list(map(ref_func, x)))
        handles.append(handle), labels.append('Reference')
    plt.legend(handles, labels)
    plt.show()
    if filename is not None:
        fig.savefig(f'{filename}.png')


if __name__ == '__main__':
    matplotlib.rcParams["figure.dpi"] = 500
    if not os.path.isdir('figures'):
        os.makedirs('figures', exist_ok=True)

    # Problem 1.2
    # Relationship between E[T_c] and p, fixing a_2 - a_1 = 4
    def e_tc_p():
        for i in range(1, 51):
            p = 0.5 + i * 0.01
            f = partial(simulate, 0, 4, p, 1 - p, 4000)
            yield p, expectation(f, 10000)
    print('Running E[T_c] - p relationship ...')
    draw(e_tc_p, 'p', 'E[T_c]', lambda p: 4 / 2 / (2 * p - 1),
         filename='figures/discrete_1d_e_tc_p')

    # Relationship between E[T_c] and a_2 - a_1, fixing p = 0.8
    def e_tc_delta_a():
        for i in range(0, 34, 2):
            f = partial(simulate, 0, i, 0.8, 0.2, 4000)
            yield i, expectation(f, 10000)
    print('Running E[T_c] - (a_2 - a_1) relationship ...')
    draw(e_tc_delta_a, 'a_2 - a_1', 'E[T_c]', lambda a: a / 2 / (2 * 0.8 - 1),
         desc='(even number)', filename='figures/discrete_1d_e_tc_a')

    # Relationship between Var[T_c] and p, fixing a_2 - a_1 = 4
    def e_var_p():
        for i in range(1, 51):
            p = 0.5 + i * 0.01
            f = partial(simulate, 0, 4, p, 1 - p, 8000)
            yield p, variance(f, 100000)
    print('Running Var[T_c] - p relationship ...')
    draw(e_var_p, 'p', 'Var[T_c]', lambda p: 4 * (1 - p) * p / ((2 * p - 1) ** 3),
         filename='figures/discrete_1d_var_tc_p')

    # Relationship between Var[T_c] and a_2 - a_1, fixing p = 0.8
    def e_var_delta_a():
        for i in range(0, 34, 2):
            f = partial(simulate, 0, i, 0.8, 0.2, 8000)
            yield i, variance(f, 100000)
    print('Running Var[T_c] - (a_2 - a_1) relationship ...')
    draw(e_var_delta_a, 'a_2 - a_1', 'Var[T_c]', lambda a: a * (1 - 0.8) * 0.8 / ((2 * 0.8 - 1) ** 3),
         desc='(even number)', filename='figures/discrete_1d_var_tc_a')

    # Problem 1.3
    # Combination number
    p13_steps_limit = 100
    c = prepare_comb(p13_steps_limit * 2)

    def get_c(m, n) -> int:
        assert p13_steps_limit * 2 >= m > 0 and n >= 0
        return c[m, n] if m >= n else 0

    # Distribution (by expression)
    def p_tc_n(a1, a2, n):
        return get_c(2 * n, n - (a2 - a1) // 2) * (0.5 ** (2 * n + 1)) * (a2 - a1) / n

    # Distribution when a_2 - a_1 = 4, p = 0.5
    def t_c_distribution():
        f = partial(simulate, 0, 4, 0.5, 0.5, p13_steps_limit, True)
        a, d = multiple_simulate(f, False, 1000000)
        return d
    print('Running discrete 1D T_c distribution ...')
    draw(t_c_distribution, 'T_c', 'Distribution', lambda n: p_tc_n(0, 4, n),
         filename='figures/discrete_1d_distribution')

    # Problem 2
    # Prepare 2D Distribution when (a_2 - a_1, b_2 - b_1) = (2, 2)
    p2_steps_limit = 100

    def prep_2d(a1: int, b1: int, a2: int, b2: int):
        f, p = {(0, 0): 1}, [0]
        for i in range(1, p2_steps_limit * 2 + 1):
            new_f = {}
            for ((x, y), v) in f.items():
                if x == a2 - a1 and y == b2 - b1:
                    continue
                for (sx, sy) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    pos = (x + sx, y + sy)
                    if pos not in new_f:
                        new_f[pos] = 0
                    new_f[pos] += v * 0.25
            f = new_f
            p.append(f[(a2 - a1, b2 - b1)] if (a2 - a1, b2 - b1) in f else 0)
        return p
    p2d = prep_2d(0, 0, 2, 2)

    # Distribution when (a_2 - a_1, b_2 - b_1) = (2, 2)
    def t_c_distribution_2d():
        f = partial(simulate_2d, (0, 0), (2, 2), p2_steps_limit)
        a, d = multiple_simulate(f, False, 1000000)
        return d
    print('Running discrete 2D T_c distribution ...')
    draw(t_c_distribution_2d, 'T_c', '2D Distribution', lambda n: p2d[2 * n],
         filename='figures/discrete_2d_distribution')

    # Problem 3.1
    # Distribution for T_c of Brownian motion when a_2 - a_1 = 1
    def brownian_t_c_distribution():
        f = partial(simulate_brownian, 0, 1, 20000, 0.0005)
        a, d = multiple_simulate(f, True, 10000)
        return d
    print('Running Brownian T_c distribution ...')
    draw(brownian_t_c_distribution, 'T_c', 'Distribution', lambda t: 2 * (1 - stats.norm.cdf(1 / math.sqrt(2 * t))),
         filename='figures/brownian_t_c_distribution')

    # Distribution for X_c of Brownian motion when a_1 = 0, a_2 = 1
    def brownian_x_c_distribution():
        f = partial(simulate_brownian, 0, 1, 200000, 0.01, return_pos=True)
        a, d = multiple_simulate(f, True, 10000, True)
        return d
    print('Running Brownian X_c distribution ...')
    draw(brownian_x_c_distribution, 'X_c', 'Distribution', lambda x: 0.5 + math.atan(2 * x - 1) / math.pi,
         filename='figures/brownian_x_c_distribution')

    # Problem 3.2
    # Guess the result
    # def guess_func(x, a, b, c, d, e):
    #     if isinstance(x, (int, float)):
    #         return 0 if x == 0 else a + e * np.arctan(b * x + c / x + d)
    #     return np.where(x > 0, a + e * np.arctan(b * x + c / x + d), 0)
    #
    # # Simulation
    # f = partial(simulate_brownian, 0, 1, 2000, 0.005, return_max=True)
    # a, d = multiple_simulate(f, True, 2000, False)
    # opt, cov = optimize.curve_fit(guess_func, [x[0] for x in d], [x[1] for x in d], maxfev=1000000)
    # print(f' > Opt results: {opt}')
    # ref = partial(guess_func, a=opt[0], b=opt[1], c=opt[2], d=opt[3], e=opt[4])

    # Distribution for Min_c of Brownian motion when a_1 = 0, a_2 = 1
    def brownian_min_c_distribution():
        f = partial(simulate_brownian, 0, 1, 1000000, 0.01, return_min=True)
        a, d = multiple_simulate(f, True, 2000, False)
        return d

    print('Running Brownian Min_c distribution ...')
    draw(brownian_min_c_distribution, 'Min_c', 'Distribution',
         ref_func=lambda x: 1 - 4 / math.pi * math.atan(x / (x - 1)),
         filename='figures/brownian_min_c_distribution')

    # Distribution for Max_c of Brownian motion when a_1 = 0, a_2 = 1
    def brownian_max_c_distribution():
        f = partial(simulate_brownian, 0, 1, 1000000, 0.01, return_max=True)
        a, d = multiple_simulate(f, True, 2000, False)
        return d

    print('Running Brownian Max_c distribution ...')
    draw(brownian_max_c_distribution, 'Max_c', 'Distribution',
         ref_func=lambda x: 0 if x == 0 else 4 / math.pi * math.atan(1 - 1 / x),
         filename='figures/brownian_max_c_distribution')
