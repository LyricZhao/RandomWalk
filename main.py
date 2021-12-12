import random
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


def expectation(bind_func_obj: partial, simulation_times: int = 100000) -> float:
    s = 0
    assert simulation_times > 0
    for i in range(simulation_times):
        x = None
        while x is None:
            x = bind_func_obj()
        s = s + x
    return s / simulation_times


def draw(func, x_label: str, y_label: str):
    x, y = [], []
    for a, b in func():
        x.append(a), y.append(b)
    plt.title(f'{y_label} towards {x_label}')
    plt.xlabel(x_label), plt.ylabel(y_label)
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    # Relationship between E[T_c] and p, fixing a_2 - a_1 = 4
    def e_tc_p():
        for i in range(1, 51):
            p = 0.5 + i * 0.01
            f = partial(simulate, 0, 4, p, 1 - p)
            yield p, expectation(f)
    draw(e_tc_p, 'p', 'E[T_c]')

    # Relationship between E[T_c] and a_2 - a_1, fixing p = 0.8
    def e_tc_delta_a():
        for i in range(0, 34, 2):
            f = partial(simulate, 0, i, 0.8, 0.2)
            yield i, expectation(f)
    draw(e_tc_delta_a, 'a_2 - a_1', 'E[T_c]')
