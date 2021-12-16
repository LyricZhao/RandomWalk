import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from functools import partial

global_steps_limit = 300


def choice(p: float) -> bool:
    assert 0 <= p <= 1
    return random.uniform(0, 1) <= p


def shift_choice(p: float) -> int:
    return 1 if choice(p) else -1


def simulate(a1: int, a2: int, p1: float, p2: float, steps_limit: int = global_steps_limit) -> int:
    if a1 == a2:
        return 0
    assert a1 <= a2, 'The point 1 must be on the left of the point 2'
    assert (a2 - a1) % 2 == 0, 'The distance between two points must be even'
    steps = 0
    for i in range(steps_limit):
        steps = steps + 1
        a1 = a1 + shift_choice(p1)
        a2 = a2 + shift_choice(p2)
        if a1 == a2:
            return steps
    return None


def multiple_simulate(bind_func_obj: partial, simulation_times: int = 100000) -> ([int], [(int, float)]):
    a, d = [], {}
    assert simulation_times > 0
    for i in range(simulation_times):
        x = None
        while x is None:
            x = bind_func_obj()
        a.append(x)
        d[x] = d[x] + 1 if x in d else 1
    d = sorted(d.items())
    s = sum(map(lambda x: x[1], d))
    d = list(map(lambda x: (x[0], x[1] / s), d))
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
    plt.title(f'{y_label} towards {x_label}')
    plt.xlabel(x_label), plt.ylabel(y_label)
    [handles, ], labels = [plt.plot(x, y)], ['Simulation']
    if ref_func is not None:
        handle, = plt.plot(x, list(map(ref_func, x)))
        handles.append(handle), labels.append('Reference')
    plt.legend(handles, labels)
    plt.show()


if __name__ == '__main__':
    matplotlib.rcParams["figure.dpi"] = 300

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
    c = prepare_comb(global_steps_limit)

    # Combination number
    def get_c(m, n) -> int:
        assert global_steps_limit >= m > 0 and n >= 0
        return c[m, n] if m >= n else 0

    # Distribution (by expression)
    def p_tc_n(a1, a2, n):
        s, p = (a2 - a1) // 2, 0
        for k in range(s, (n + s) // 2 + 1):
            p += (0.5 ** (n + 2 * k - s)) * \
                 get_c(n - 1, n - (2 * k - s)) * (get_c(2 * k - s - 1, k - 1) - get_c(2 * k - s - 1, k))
        return p

    # Distribution when a_2 - a_1 = 4, p = 0.5
    def t_c_distribution():
        f = partial(simulate, 0, 4, 0.5, 0.5)
        a, d = multiple_simulate(f)
        return d
    draw(t_c_distribution, 'T_c', 'Distribution', lambda n: p_tc_n(0, 4, n))
