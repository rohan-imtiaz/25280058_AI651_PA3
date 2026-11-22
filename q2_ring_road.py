import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

number_of_sensors = 40


def true_step(snapshot):
    # Backward motion: x_i(t + 1) = x_{i + 1}(t)
    return np.roll(snapshot, -1)


def part_a_one():
    print("Q2(a)(i): least squares on the isolated slowdown")
    snapshot = np.zeros(number_of_sensors)
    snapshot[20] = 1
    target = true_step(snapshot)
    # Each row is one sensor with features [x_i, x_{i - 1} + x_{i + 1}]
    features = np.stack([snapshot, np.roll(snapshot, 1) + np.roll(snapshot, -1)], axis = 1)
    solution, _, _, _ = np.linalg.lstsq(features, target, rcond = None)
    a, b = solution
    prediction = features @ solution
    loss = np.sum((prediction - target) ** 2)
    print("a =", round(a, 6), "b =", round(b, 6), "loss =", round(loss, 6))
    print("Sensors 19, 20, 21 predicted:", prediction[19:22], "correct:", target[19:22])
    return a, b


def apply_symmetric_layer(snapshot, a, b):
    return a * snapshot + b * (np.roll(snapshot, 1) + np.roll(snapshot, -1))


def part_a_two(a, b):
    print("Q2(a)(ii): rolling out the fitted layer on a smooth bump")
    sensor_index = np.arange(number_of_sensors)
    initial = np.exp(-((sensor_index - 20) ** 2) / 8)
    predicted_sequence = [initial]
    true_sequence = [initial]
    for step in range(20):
        predicted_sequence.append(apply_symmetric_layer(predicted_sequence[-1], a, b))
        true_sequence.append(true_step(true_sequence[-1]))
    predicted_sequence = np.array(predicted_sequence)
    true_sequence = np.array(true_sequence)
    print("t = 20 peak position, predicted:", int(predicted_sequence[-1].argmax()), "true:", int(true_sequence[-1].argmax()))
    print("t = 20 peak height, predicted:", round(predicted_sequence[-1].max(), 4), "true:", round(true_sequence[-1].max(), 4))

    maximum_value = max(predicted_sequence.max(), true_sequence.max())
    figure, axes = plt.subplots(1, 2, figsize = (11, 5), sharey = True)
    titles = ["Predicted by the symmetric layer", "True motion"]
    for axis, sequence, title in zip(axes, [predicted_sequence, true_sequence], titles):
        image = axis.imshow(sequence, aspect = "auto", origin = "lower", cmap = "viridis", vmin = 0, vmax = maximum_value)
        axis.set_title(title)
        axis.set_xlabel("Sensor index")
    axes[0].set_ylabel("Time t")
    figure.colorbar(image, ax = axes.tolist(), label = "Slowdown")
    figure.savefig(os.path.join(config.figure_directory, "q2a_heatmaps.png"), dpi = 150, bbox_inches = "tight")
    plt.close(figure)


def part_b():
    print("Q2(b): direction aware weights")
    random_generator = np.random.default_rng(0)
    a, weight_behind, weight_ahead = 0, 0, 1
    for trial in range(5):
        snapshot = random_generator.normal(size = number_of_sensors)
        prediction = a * snapshot + weight_behind * np.roll(snapshot, 1) + weight_ahead * np.roll(snapshot, -1)
        if not np.allclose(prediction, true_step(snapshot)):
            raise ValueError("Direction aware layer failed")
    print("a = 0, w_minus = 0, w_plus = 1 matches the true step on random snapshots")

    # A 1D convolution (cross correlation) with kernel [0, 0, 1] on a circularly padded ring
    kernel = np.array([0, 0, 1])
    snapshot = random_generator.normal(size = number_of_sensors)
    padded = np.concatenate([snapshot[-1:], snapshot, snapshot[:1]]) # Circular padding of 1
    convolved = np.array([np.dot(padded[i:i + 3], kernel) for i in range(number_of_sensors)])
    print("Kernel [0, 0, 1] with circular padding matches np.roll(x, -1):", np.allclose(convolved, true_step(snapshot)))
    # In PyTorch this is F.pad(x, (1, 1), mode = "circular") followed by F.conv1d with weight [[[0, 0, 1]]]


def history_layer(current, previous, a, b, c):
    return a * current + b * (np.roll(current, 1) + np.roll(current, -1)) + c * previous


def part_c():
    print("Q2(c): using history with equal neighbour weights")
    random_generator = np.random.default_rng(1)
    a, b, c = 0, 1, -1
    for direction_name, step_function in [("backward", lambda x: np.roll(x, -1)), ("forward", lambda x: np.roll(x, 1))]:
        for trial in range(5):
            previous = random_generator.normal(size = number_of_sensors)
            current = step_function(previous)
            correct_next = step_function(current)
            prediction = history_layer(current, previous, a, b, c)
            if not np.allclose(prediction, correct_next):
                raise ValueError("History layer failed for " + direction_name + " motion")
        print("a = 0, b = 1, c = -1 is exact for " + direction_name + " motion")


if __name__ == "__main__":
    config.make_directories()
    fitted_a, fitted_b = part_a_one()
    part_a_two(fitted_a, fitted_b)
    part_b()
    part_c()