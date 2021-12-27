import math

import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from functools import partial
from scipy import stats

global_steps_limit = 400
global_simulate_times = 100000
global_brownian_delta_t = 0.01


def choice(p: float) -> bool:
    assert 0 <= p <= 1
    return random.uniform(0, 1) <= p


def shift_choice(p: float) -> int:
    return 1 if choice(p) else -1


def simulate(a1: int, a2: int, p1: float, p2: float, steps_limit: int = global_steps_limit) -> int:
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
    return None


def simulate_brownian(a1: float, a2: float,
                      steps_limits: int = global_steps_limit,
                      delta_t: float = global_brownian_delta_t) -> float:
    if a1 == a2:
        return 0
    sign = (a1 < a2)
    steps = 0
    sqrt_delta_t = math.sqrt(delta_t)
    for i in range(steps_limits):
        steps = steps + 1
        a1 = a1 + sqrt_delta_t * random.gauss(0, 1)
        a2 = a2 + sqrt_delta_t * random.gauss(0, 1)
        if sign != (a1 < a2):
            return steps * delta_t
    return None


def simulate_hit_brownian(a: float,
                          steps_limits: int = global_steps_limit,
                          delta_t: float = global_brownian_delta_t) -> float:
    steps = 0
    sign = (a > 0)
    sqrt_delta_t = math.sqrt(delta_t)
    for i in range(steps_limits):
        steps = steps + 1
        a = a + sqrt_delta_t * np.random.randn()
        if sign != (a > 0):
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


def multiple_simulate(bind_func_obj: partial, accumulate: bool = False, simulation_times: int = global_simulate_times) -> ([int], [(int, float)]):
    a, d = [], {}
    assert simulation_times > 0
    for i in range(simulation_times):
        x = bind_func_obj()
        if x is not None:
            a.append(x)
            d[x] = d[x] + 1 if x in d else 1
    d = sorted(d.items())
    d = list(map(lambda x: (x[0], x[1] / simulation_times), d))
    if accumulate:
        for i in range(1, len(d)):
            d[i] = (d[i][0], d[i - 1][1] + d[i][1])
    return a, d


def expectation(bind_func_obj: partial) -> float:
    a, _ = multiple_simulate(bind_func_obj)
    return np.mean(a)


def variance(bind_func_obj: partial) -> float:
    a, _ = multiple_simulate(bind_func_obj)
    return np.var(a, ddof=1)  # should be without bias


def prepare_comb(n: int) -> np.array:
    c = np.zeros((n + 1, n + 1), dtype=float)
    for i in range(n + 1):
        c[i, 0] = 1
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            c[i, j] = c[i - 1, j - 1] + c[i - 1, j]
    return c


def draw(func, x_label: str, y_label: str, ref_func=None):
    x, y = [], []
    for a, b in func():
        x.append(a), y.append(b)
    fig = plt.figure()
    plt.title(f'{y_label} towards {x_label}')
    plt.xlabel(x_label), plt.ylabel(y_label)
    [handles, ], labels = [plt.plot(x, y)], ['Simulation']
    if ref_func is not None:
        handle, = plt.plot(x, list(map(ref_func, x)))
        handles.append(handle), labels.append('Reference')
    for i in range(len(y)):
        y[i] = y[i] - ref_func(x[i])
    plt.plot(x, y)
    plt.legend(handles, labels)
    plt.show()
    # fig.savefig(f'{y_label} towards {x_label}.png')


if __name__ == '__main__':
    matplotlib.rcParams["figure.dpi"] = 500

    # Distribution of Brownian motion when a_2 - a_1 = 1
    def brownian_tc_distribution():
        f = partial(simulate_hit_brownian, 1)
        a, d = multiple_simulate(f, True)
        return d
    draw(brownian_tc_distribution, 'T_c', 'Distribution', lambda t: 2 * (1 - stats.norm.cdf(1, scale=math.sqrt(t))))
    exit(0)

    # Relationship between E[T_c] and p, fixing a_2 - a_1 = 4
    def e_tc_p():
        for i in range(1, 51):
            p = 0.5 + i * 0.01
            f = partial(simulate, 0, 4, p, 1 - p)
            yield p, expectation(f)
    draw(e_tc_p, 'p', 'E[T_c]', lambda p: 4 / 2 / (2 * p - 1))

    # Relationship between E[T_c] and a_2 - a_1, fixing p = 0.8
    def e_tc_delta_a():
        for i in range(0, 34, 2):
            f = partial(simulate, 0, i, 0.8, 0.2)
            yield i, expectation(f)
    draw(e_tc_delta_a, 'a_2 - a_1', 'E[T_c]', lambda a: a / 2 / (2 * 0.8 - 1))

    # Relationship between Var[T_c] and p, fixing a_2 - a_1 = 4
    def e_var_p():
        for i in range(1, 51):
            p = 0.5 + i * 0.01
            f = partial(simulate, 0, 4, p, 1 - p)
            yield p, variance(f)
    draw(e_var_p, 'p', 'Var[T_c]', lambda p: 4 * (1 - p) * p / ((2 * p - 1) ** 3))

    # Relationship between Var[T_c] and a_2 - a_1, fixing p = 0.8
    def e_var_delta_a():
        for i in range(0, 34, 2):
            f = partial(simulate, 0, i, 0.8, 0.2)
            yield i, variance(f)
    draw(e_var_delta_a, 'a_2 - a_1', 'Var[T_c]', lambda a: a * (1 - 0.8) * 0.8 / ((2 * 0.8 - 1) ** 3))

    # Combination number
    c = prepare_comb(global_steps_limit * 2)

    # Combination number
    def get_c(m, n) -> int:
        assert global_steps_limit * 2 >= m > 0 and n >= 0
        return c[m, n] if m >= n else 0

    # Distribution (by expression)
    def p_tc_n(a1, a2, n):
        return get_c(2 * n, n - (a2 - a1) // 2) * (0.5 ** (2 * n + 1)) * (a2 - a1) / n

    # Distribution when a_2 - a_1 = 4, p = 0.5
    def t_c_distribution():
        f = partial(simulate, 0, 4, 0.5, 0.5)
        a, d = multiple_simulate(f)
        return d
    draw(t_c_distribution, 'T_c', 'Distribution', lambda n: p_tc_n(0, 4, n))

    # Prepare 2D Distribution when (a_2 - a_1, b_2 - b_1) = (2, 2)
    def prep_2d(a1: int, b1: int, a2: int, b2: int):
        f, p = {(0, 0): 1}, [0]
        for i in range(1, global_steps_limit * 2 + 1):
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

    # Distribution
    def t_c_distribution_2d():
        f = partial(simulate_2d, (0, 0), (2, 2))
        a, d = multiple_simulate(f)
        return d
    draw(t_c_distribution_2d, 'T_c', '2D Distribution', lambda n: p2d[2 * n])
