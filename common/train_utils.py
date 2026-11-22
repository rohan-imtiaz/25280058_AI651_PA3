import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

import config
from common import data_utils
from common.models import SpatioTemporalModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_hardware_description():
    description = {"cpu_count": os.cpu_count(), "torch_version": torch.__version__}
    if torch.cuda.is_available():
        description["gpu"] = torch.cuda.get_device_name(0)
    else:
        description["gpu"] = "none"
    return description


def synchronise(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def evaluate(model, segment, mean, std, device, sensor_indices = None):
    model.eval()
    total_windows = data_utils.count_windows(segment)
    number_of_nodes = segment.shape[1]
    absolute_sum = 0
    squared_sum = 0
    horizon_absolute = torch.zeros(config.output_length, device = device, dtype = torch.float64)
    kept_predictions = []
    kept_targets = []
    with torch.no_grad():
        for start in range(0, total_windows, config.evaluation_batch_size):
            stop = min(start + config.evaluation_batch_size, total_windows)
            starts = torch.arange(start, stop, device = device)
            x, y = data_utils.make_batch(segment, starts)
            prediction = model(x) * std + mean # Back to original flow units
            target = y * std + mean
            error = (prediction - target).double()
            absolute_sum = absolute_sum + error.abs().sum().item()
            squared_sum = squared_sum + (error ** 2).sum().item()
            horizon_absolute = horizon_absolute + error.abs().sum(dim = (0, 2))
            if sensor_indices is not None:
                kept_predictions.append(prediction[:, :, sensor_indices].cpu())
                kept_targets.append(target[:, :, sensor_indices].cpu())
    count = total_windows * config.output_length * number_of_nodes
    metrics = {
        "mae": absolute_sum / count,
        "rmse": (squared_sum / count) ** 0.5,
        "mae_per_horizon": (horizon_absolute / (total_windows * number_of_nodes)).cpu().tolist(),
    }
    if sensor_indices is not None:
        metrics["predictions"] = torch.cat(kept_predictions).numpy()
        metrics["targets"] = torch.cat(kept_targets).numpy()
    return metrics


def train_model(model_name, adjacency_mode, fixed_adjacency, normalised_segments, mean, std, device, number_of_epochs):
    set_seed(config.seed)
    number_of_nodes = normalised_segments["train"].shape[1]
    model = SpatioTemporalModel(
        adjacency_mode, fixed_adjacency, number_of_nodes, config.input_length, config.output_length,
        config.hidden_dimension, config.embedding_dimension, config.top_k, config.learn_alpha,
        config.use_skip_connection,
    ).to(device)
    number_of_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print("Model " + model_name + " has " + str(number_of_parameters) + " trainable parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)
    loss_function = nn.L1Loss()
    train_segment = normalised_segments["train"]
    train_windows = data_utils.count_windows(train_segment)

    best_validation_mae = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    synchronise(device)
    start_time = time.time()
    for epoch in range(number_of_epochs):
        model.train()
        order = torch.randperm(train_windows, device = device)
        loss_total = torch.zeros(1, device = device)
        batch_count = 0
        for start in range(0, train_windows, config.batch_size):
            x, y = data_utils.make_batch(train_segment, order[start:start + config.batch_size])
            optimizer.zero_grad()
            loss = loss_function(model(x), y)
            loss.backward()
            optimizer.step()
            loss_total = loss_total + loss.detach()
            batch_count = batch_count + 1
        validation = evaluate(model, normalised_segments["validation"], mean, std, device)
        train_loss = loss_total.item() / batch_count
        history.append({"epoch": epoch + 1, "train_loss_normalised": train_loss, "validation_mae": validation["mae"]})
        print("Epoch", epoch + 1, "train loss", round(train_loss, 4), "validation MAE", round(validation["mae"], 3))
        if validation["mae"] < best_validation_mae:
            best_validation_mae = validation["mae"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement = epochs_without_improvement + 1
            if epochs_without_improvement >= config.patience:
                print("Early stopping at epoch", epoch + 1)
                break
    synchronise(device)
    training_seconds = time.time() - start_time # Includes the validation pass after every epoch

    model.load_state_dict(best_state)
    validation = evaluate(model, normalised_segments["validation"], mean, std, device)
    test = evaluate(model, normalised_segments["test"], mean, std, device, config.plot_sensor_indices)

    result = {
        "model_name": model_name,
        "adjacency_mode": adjacency_mode,
        "validation_mae": validation["mae"],
        "validation_rmse": validation["rmse"],
        "test_mae": test["mae"],
        "test_rmse": test["rmse"],
        "test_mae_per_horizon": test["mae_per_horizon"],
        "training_seconds": training_seconds,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "number_of_parameters": number_of_parameters,
        "use_skip_connection": config.use_skip_connection,
        "history": history,
    }
    if adjacency_mode == "hybrid":
        result["final_alpha"] = float(model.get_alpha().item())
    save_outputs(model_name, adjacency_mode, model, result, test)
    return result


def save_outputs(model_name, adjacency_mode, model, result, test):
    with open(os.path.join(config.result_directory, model_name + "_result.json"), "w") as file:
        json.dump(result, file, indent = 2)
    np.savez(
        os.path.join(config.result_directory, model_name + "_test_predictions.npz"),
        predictions = test["predictions"],
        targets = test["targets"],
    )
    model.eval()
    with torch.no_grad():
        if adjacency_mode in ["adaptive", "hybrid"]:
            np.save(os.path.join(config.result_directory, model_name + "_adaptive_adjacency.npy"), model.get_adaptive_adjacency().cpu().numpy())
        if adjacency_mode != "none":
            np.save(os.path.join(config.result_directory, model_name + "_final_adjacency.npy"), model.get_adjacency().cpu().numpy())