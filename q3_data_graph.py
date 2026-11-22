import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from common import data_utils, graph_utils


def plot_week(flow, timestamps, sensor_ids):
    week_start = pd.Timestamp(config.plot_week_start)
    mask = (timestamps >= week_start) & (timestamps < week_start + pd.Timedelta(days = 7))
    figure, axes = plt.subplots(len(config.plot_sensor_indices), 1, figsize = (12, 7), sharex = True)
    for axis, sensor_index in zip(axes, config.plot_sensor_indices):
        axis.plot(timestamps[mask], flow[mask, sensor_index])
        axis.set_ylabel("Flow per 15 min")
        axis.set_title("Sensor " + str(sensor_ids[sensor_index]) + " (index " + str(sensor_index) + ")")
    axes[-1].set_xlabel("Time")
    figure.tight_layout()
    figure.savefig(os.path.join(config.figure_directory, "q3a_three_sensors_week.png"), dpi = 150)
    plt.close(figure)


def graph_statistics(adjacency):
    summary = {}
    summary["number_of_nodes"] = int(adjacency.shape[0])
    summary["nonzero_entries_total"] = int(np.count_nonzero(adjacency))
    summary["nonzero_diagonal_entries"] = int(np.count_nonzero(np.diag(adjacency)))
    off_diagonal = graph_utils.remove_self_loops(adjacency)
    summary["nonzero_edges_off_diagonal"] = int(np.count_nonzero(off_diagonal))
    summary["symmetric"] = graph_utils.is_symmetric(adjacency)
    summary["minimum_weight"] = float(adjacency[adjacency != 0].min())
    summary["maximum_weight"] = float(adjacency.max())
    degrees = np.count_nonzero(off_diagonal, axis = 1)
    summary["degree_min"] = int(degrees.min())
    summary["degree_mean"] = float(degrees.mean())
    summary["degree_median"] = float(np.median(degrees))
    summary["degree_max"] = int(degrees.max())
    summary["isolated_nodes"] = int((degrees == 0).sum())
    print(json.dumps(summary, indent = 2))

    figure, axis = plt.subplots(figsize = (6, 5))
    image = axis.imshow(adjacency, cmap = "viridis")
    axis.set_title("Road distance adjacency (716 x 716)")
    axis.set_xlabel("Sensor index")
    axis.set_ylabel("Sensor index")
    figure.colorbar(image, ax = axis)
    figure.savefig(os.path.join(config.figure_directory, "q3b_adjacency_heatmap.png"), dpi = 150, bbox_inches = "tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize = (6, 4))
    axis.hist(degrees, bins = 30)
    axis.set_xlabel("Number of neighbours")
    axis.set_ylabel("Number of sensors")
    axis.set_title("Neighbour count distribution")
    figure.savefig(os.path.join(config.figure_directory, "q3b_degree_distribution.png"), dpi = 150, bbox_inches = "tight")
    plt.close(figure)
    return summary, degrees


def neighbourhoods(adjacency, degrees, metadata):
    binary = (graph_utils.remove_self_loops(adjacency) != 0).astype(int)
    # The sensor with median degree is used so the example is typical
    center = int(np.argmin(np.abs(degrees - np.median(degrees))))
    one_hop = np.where(binary[center] > 0)[0]
    reach_two = ((binary @ binary)[center] > 0) | (binary[center] > 0) # Paths with at most two edges
    reach_two[center] = False
    two_hop_only = np.setdiff1d(np.where(reach_two)[0], one_hop)
    print("Selected sensor index:", center, "ID:", metadata["ID"][center])
    print("One hop neighbours (" + str(len(one_hop)) + "):", one_hop.tolist())
    print("Two hop only neighbours (" + str(len(two_hop_only)) + "):", two_hop_only.tolist())

    figure, axis = plt.subplots(figsize = (7, 6))
    axis.scatter(metadata["Lng"], metadata["Lat"], s = 6, color = "lightgrey", label = "Other sensors")
    axis.scatter(metadata["Lng"][two_hop_only], metadata["Lat"][two_hop_only], s = 14, color = "orange", label = "Two hop only")
    axis.scatter(metadata["Lng"][one_hop], metadata["Lat"][one_hop], s = 20, color = "green", label = "One hop")
    axis.scatter(metadata["Lng"][center], metadata["Lat"][center], s = 60, color = "red", label = "Selected sensor")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.legend()
    figure.savefig(os.path.join(config.figure_directory, "q3c_neighbourhoods.png"), dpi = 150, bbox_inches = "tight")
    plt.close(figure)


if __name__ == "__main__":
    config.make_directories()
    metadata, adjacency = data_utils.load_san_diego_metadata_and_adjacency()
    sensor_ids = metadata["ID"].astype(str).tolist()
    filled = data_utils.prepare_flow(sensor_ids)
    data_utils.save_cache(filled, adjacency, metadata)
    print("Flow array shape (time, nodes):", filled.shape)
    print("Time range:", filled.index[0], "to", filled.index[-1])

    plot_week(filled.values, filled.index, sensor_ids)
    summary, degrees = graph_statistics(adjacency)
    neighbourhoods(adjacency, degrees, metadata)
    with open(os.path.join(config.result_directory, "q3_graph_summary.json"), "w") as file:
        json.dump(summary, file, indent = 2)