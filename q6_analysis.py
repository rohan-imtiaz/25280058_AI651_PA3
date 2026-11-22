import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from common import data_utils, graph_utils

model_names = ["temporal", "fixed", "adaptive", "hybrid"]
display_names = {"temporal": "Temporal-only", "fixed": "STGNN Fixed", "adaptive": "STGNN Adaptive", "hybrid": "STGNN Hybrid"}
adjacency_labels = {"temporal": "None", "fixed": "Fixed", "adaptive": "Adaptive", "hybrid": "Hybrid"}
strongest_per_row = 5


def load_result(model_name):
    with open(os.path.join(config.result_directory, model_name + "_result.json")) as file:
        return json.load(file)


def results_table():
    rows = []
    for model_name in model_names:
        result = load_result(model_name)
        rows.append({
            "Model": display_names[model_name],
            "Adjacency": adjacency_labels[model_name],
            "Val MAE": round(result["validation_mae"], 3),
            "Val RMSE": round(result["validation_rmse"], 3),
            "Test MAE": round(result["test_mae"], 3),
            "Test RMSE": round(result["test_rmse"], 3),
            "Training time (s)": round(result["training_seconds"], 1),
            "Epochs run": result["epochs_run"],
        })
        if model_name == "hybrid":
            print("Final learned alpha:", round(result["final_alpha"], 4))
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(config.result_directory, "final_table.csv"), index = False)
    print(table.to_string(index = False))


def prediction_plots():
    horizon = config.plot_horizon_index
    steps_in_week = 7 * 96
    figure, axes = plt.subplots(len(config.plot_sensor_indices), 1, figsize = (12, 8), sharex = True)
    for model_name in model_names:
        data = np.load(os.path.join(config.result_directory, model_name + "_test_predictions.npz"))
        for column, axis in enumerate(axes):
            if model_name == model_names[0]:
                axis.plot(data["targets"][:steps_in_week, horizon, column], color = "black", label = "Actual")
            axis.plot(data["predictions"][:steps_in_week, horizon, column], label = display_names[model_name], alpha = 0.8)
    for axis, sensor_index in zip(axes, config.plot_sensor_indices):
        axis.set_title("Sensor index " + str(sensor_index) + ", horizon " + str((horizon + 1) * 15) + " min, first test week")
        axis.set_ylabel("Flow per 15 min")
    axes[0].legend(ncol = 5)
    axes[-1].set_xlabel("Test window index (15 min steps)")
    figure.tight_layout()
    figure.savefig(os.path.join(config.figure_directory, "q6a_predicted_vs_actual.png"), dpi = 150)
    plt.close(figure)


def haversine_kilometres(latitude, longitude):
    latitude_radians = np.radians(latitude)
    longitude_radians = np.radians(longitude)
    delta_latitude = latitude_radians[:, None] - latitude_radians[None, :]
    delta_longitude = longitude_radians[:, None] - longitude_radians[None, :]
    inner = np.sin(delta_latitude / 2) ** 2 + np.cos(latitude_radians)[:, None] * np.cos(latitude_radians)[None, :] * np.sin(delta_longitude / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(inner))


def graph_analysis():
    _, _, fixed_adjacency, metadata = data_utils.load_cache()
    learned = np.load(os.path.join(config.result_directory, "adaptive_adaptive_adjacency.npy"))
    number_of_nodes = learned.shape[0]
    road_edges = graph_utils.remove_self_loops(fixed_adjacency) != 0
    road_edges_either_direction = road_edges | road_edges.T
    reach_two = (road_edges_either_direction.astype(int) @ road_edges_either_direction.astype(int)) > 0
    reach_two = reach_two | road_edges_either_direction
    np.fill_diagonal(reach_two, False)
    distances = haversine_kilometres(metadata["Lat"].values, metadata["Lng"].values)

    off_diagonal = graph_utils.remove_self_loops(learned)
    learned_edges = off_diagonal > 0
    strongest_edges = np.zeros_like(learned_edges)
    top_indices = np.argsort(-off_diagonal, axis = 1)[:, :strongest_per_row]
    for row in range(number_of_nodes):
        strongest_edges[row, top_indices[row]] = True

    summary = {}
    summary["road_density"] = float(road_edges.sum() / (number_of_nodes * (number_of_nodes - 1)))
    for name, edges in [("all_learned", learned_edges), ("strongest_learned", strongest_edges)]:
        summary[name + "_edge_count"] = int(edges.sum())
        summary[name + "_in_road_graph"] = float((edges & road_edges_either_direction).sum() / edges.sum())
        summary[name + "_within_two_hops"] = float((edges & reach_two).sum() / edges.sum())
        summary[name + "_mean_distance_km"] = float(distances[edges].mean())
        summary[name + "_median_distance_km"] = float(np.median(distances[edges]))
        summary[name + "_reverse_edge_present"] = float((edges & edges.T).sum() / edges.sum())
    summary["road_mean_distance_km"] = float(distances[road_edges].mean())
    all_pairs = ~np.eye(number_of_nodes, dtype = bool)
    summary["all_pairs_mean_distance_km"] = float(distances[all_pairs].mean())
    print(json.dumps(summary, indent = 2))
    with open(os.path.join(config.result_directory, "q6b_graph_summary.json"), "w") as file:
        json.dump(summary, file, indent = 2)

    figure, axes = plt.subplots(1, 2, figsize = (11, 5))
    axes[0].imshow(road_edges.astype(float), cmap = "viridis")
    axes[0].set_title("Road graph (nonzero entries)")
    axes[1].imshow(learned, cmap = "viridis")
    axes[1].set_title("Learned adaptive adjacency")
    for axis in axes:
        axis.set_xlabel("Sensor index")
        axis.set_ylabel("Sensor index")
    figure.savefig(os.path.join(config.figure_directory, "q5b_learned_vs_road_adjacency.png"), dpi = 150, bbox_inches = "tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize = (7, 4))
    axis.hist(distances[all_pairs], bins = 50, density = True, alpha = 0.4, label = "All pairs")
    axis.hist(distances[road_edges], bins = 50, density = True, alpha = 0.5, label = "Road graph edges")
    axis.hist(distances[learned_edges], bins = 50, density = True, alpha = 0.5, label = "Learned edges")
    axis.set_xlabel("Straight line distance (km)")
    axis.set_ylabel("Density")
    axis.legend()
    figure.savefig(os.path.join(config.figure_directory, "q6b_edge_distances.png"), dpi = 150, bbox_inches = "tight")
    plt.close(figure)


if __name__ == "__main__":
    config.make_directories()
    results_table()
    prediction_plots()
    graph_analysis()