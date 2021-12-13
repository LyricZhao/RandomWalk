import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from functools import partial


def choice(p: float) -> bool:
    assert 0 <= p <= 1
    return random.uniform(0, 1) <= p


def shift_choice(p: float) -> int:
    return 1 if choice(p) else -1


def simulate(a1: int, a2: int, p1: float, p2: float, steps_limit: int = 1000) -> int:
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


def multiple_simulate(bind_func_obj: partial, simulation_times: int = 100000) -> [int]:
    s = []
    assert simulation_times > 0
    for i in range(simulation_times):
        x = None
        while x is None:
            x = bind_func_obj()
        s.append(x)
    return s


def expectation(bind_func_obj: partial, simulation_times: int = 100000) -> float:
    return np.mean(multiple_simulate(bind_func_obj, simulation_times))


def variance(bind_func_obj: partial, simulation_times: int = 100000) -> float:
    return np.var(multiple_simulate(bind_func_obj, simulation_times), ddof=1)  # should be without bias


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
