import itertools
from fractions import Fraction

import numpy as np

maximum_neighbours = 8


def part_a():
    print("Q1(a): affine message")
    random_generator = np.random.default_rng(0)
    neighbourhood_one = np.array([-1, 1, 0, 0]) # 2 steady neighbours
    neighbourhood_two = np.array([0, 0, 0, 0]) # 4 steady neighbours
    for trial in range(3):
        weight = random_generator.normal()
        constant = random_generator.normal()
        sum_one = np.sum(weight * neighbourhood_one + constant)
        sum_two = np.sum(weight * neighbourhood_two + constant)
        # The sum equals w * S + c * n, so only n and S matter
        formula_one = weight * neighbourhood_one.sum() + constant * len(neighbourhood_one)
        print("w =", round(weight, 3), "c =", round(constant, 3), "aggregated:", round(sum_one, 6), round(sum_two, 6), "formula:", round(formula_one, 6))
        if not np.isclose(sum_one, sum_two):
            raise ValueError("Aggregated messages should match")
    print("Both neighbourhoods have n = 4 and S = 0, so any rho gives the same output")


def build_neighbourhood(counts):
    slowing, steady, speeding = counts
    return [-1] * slowing + [0] * steady + [1] * speeding


def all_count_triples():
    triples = []
    for slowing, steady, speeding in itertools.product(range(maximum_neighbours + 1), repeat = 3):
        total = slowing + steady + speeding
        if 1 <= total <= maximum_neighbours:
            triples.append((slowing, steady, speeding))
    return triples


def part_b():
    print("Q1(b): scalar message with place values")
    base = maximum_neighbours + 1 # Every count is at most K so a digit never reaches the base
    message = {-1: 1, 0: base, 1: base ** 2}
    print("phi(-1), phi(0), phi(1) =", message[-1], message[0], message[1])
    checked = 0
    for counts in all_count_triples():
        total = sum(message[value] for value in build_neighbourhood(counts))
        slowing_decoded = total % base
        remainder = total // base
        steady_decoded = remainder % base
        speeding_decoded = remainder // base
        if (slowing_decoded, steady_decoded, speeding_decoded) != counts:
            raise ValueError("Decoding failed for " + str(counts))
        checked = checked + 1
    print("Decoded all " + str(checked) + " triples correctly for K = " + str(maximum_neighbours))

    non_steady_message = {-1: 1, 0: 0, 1: 1}
    for counts in all_count_triples():
        total = sum(non_steady_message[value] for value in build_neighbourhood(counts))
        if total != counts[0] + counts[2]:
            raise ValueError("Non steady count failed for " + str(counts))
    print("Non steady count message phi(-1) = 1, phi(0) = 0, phi(1) = 1 works for every neighbourhood")


def aggregate(method, counts):
    slowing, steady, speeding = counts
    total = slowing + steady + speeding
    if method == "Sum":
        return (slowing, steady, speeding)
    if method == "Mean":
        return (Fraction(slowing, total), Fraction(steady, total), Fraction(speeding, total))
    return (int(slowing > 0), int(steady > 0), int(speeding > 0))


def format_vector(vector):
    return "(" + ", ".join(str(entry) for entry in vector) + ")"


def part_c():
    print("Q1(c): aggregation table")
    tasks = {
        "(i) Count steady neighbours": lambda counts: counts[1],
        "(ii) More than half slowing": lambda counts: counts[0] * 2 > sum(counts),
        "(iii) Any slowing": lambda counts: counts[0] > 0,
    }
    for task_name, answer_function in tasks.items():
        for method in ["Sum", "Mean", "Max"]:
            seen = {}
            collision = None
            for counts in all_count_triples():
                vector = aggregate(method, counts)
                answer = answer_function(counts)
                if vector in seen and seen[vector][1] != answer:
                    collision = (seen[vector], (counts, answer), vector)
                    break
                seen[vector] = (counts, answer)
            if collision is None:
                print(task_name, "|", method, "| Yes")
            else:
                first, second, vector = collision
                print(task_name, "|", method, "| No | vector", format_vector(vector),
                      "| neighbourhood", build_neighbourhood(first[0]), "answer", first[1],
                      "| neighbourhood", build_neighbourhood(second[0]), "answer", second[1])


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()