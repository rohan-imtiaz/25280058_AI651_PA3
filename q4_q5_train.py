import argparse
import json
import os

import torch

import config
from common import data_utils, graph_utils, train_utils

model_modes = {
    "temporal": "none",
    "fixed": "fixed",
    "adaptive": "adaptive",
    "hybrid": "hybrid",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default = "temporal,fixed,adaptive,hybrid")
    parser.add_argument("--epochs", type = int, default = config.number_of_epochs)
    arguments = parser.parse_args()

    config.make_directories()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hardware = train_utils.get_hardware_description()
    print("Device:", device, hardware)
    with open(os.path.join(config.result_directory, "hardware.json"), "w") as file:
        json.dump(hardware, file, indent = 2)

    flow, timestamps, adjacency, metadata = data_utils.load_cache()
    normalised_segments, mean, std = data_utils.split_and_normalise(flow, timestamps, device)
    _, normalisation_type = graph_utils.normalise_fixed_adjacency(adjacency)
    print("Fixed adjacency normalisation used:", normalisation_type)

    for model_name in arguments.models.split(","):
        result = train_utils.train_model(
            model_name, model_modes[model_name], adjacency, normalised_segments, mean, std, device, arguments.epochs,
        )
        print("Finished " + model_name + ": test MAE " + str(round(result["test_mae"], 3)) +
              " test RMSE " + str(round(result["test_rmse"], 3)) +
              " time " + str(round(result["training_seconds"], 1)) + " s")