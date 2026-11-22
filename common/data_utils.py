import glob
import os

import numpy as np
import pandas as pd
import torch

import config


def find_file(file_name):
    pattern = os.path.join(config.raw_data_root, "**", file_name)
    matches = glob.glob(pattern, recursive = True)
    if len(matches) == 0:
        raise FileNotFoundError("Could not find " + file_name + " under " + config.raw_data_root)
    return matches[0]


def load_san_diego_metadata_and_adjacency():
    metadata = pd.read_csv(find_file("ca_meta.csv"))
    adjacency = np.load(find_file("ca_rn_adj.npy"))
    if adjacency.shape[0] != len(metadata):
        raise ValueError("Adjacency size and metadata length do not match")
    print("Sensors per district:")
    print(metadata["District"].value_counts().sort_index().to_string())
    selected = np.where(metadata["District"] == config.san_diego_district)[0]
    print("Selected San Diego sensors:", len(selected))
    if len(selected) != config.expected_number_of_nodes:
        print("WARNING: expected " + str(config.expected_number_of_nodes) + " sensors, check the district filter")
    san_diego_metadata = metadata.iloc[selected].reset_index(drop = True)
    # The same index list is used for rows and columns so the ordering matches the metadata
    san_diego_adjacency = adjacency[np.ix_(selected, selected)]
    return san_diego_metadata, san_diego_adjacency


def load_year_flow(year, sensor_ids):
    # The raw CA file holds all 8600 sensors, so this step needs a few GB of RAM
    frame = pd.read_hdf(find_file("ca_his_raw_" + str(year) + ".h5"))
    frame.index = pd.to_datetime(frame.index)
    frame.columns = frame.columns.astype(str)
    frame = frame[frame.index.year == year]
    absent = [sensor for sensor in sensor_ids if sensor not in frame.columns]
    if len(absent) > 0:
        print("WARNING: " + str(len(absent)) + " sensors missing in the " + str(year) + " file")
    frame = frame.reindex(columns = sensor_ids).astype("float32")
    return frame


def resample_to_15_minutes(frame):
    # Flow is a vehicle count per interval, so three 5 minute steps are summed not averaged
    # min_count = 3 keeps a NaN when any of the three steps is missing
    return frame.resample("15min").sum(min_count = 3)


def fill_missing_values(combined):
    # Forward fill only uses past values, remaining gaps at the start use the 2020 mean of the sensor
    train_means = combined[combined.index.year == 2020].mean(axis = 0)
    train_means = train_means.fillna(train_means.mean())
    filled = combined.ffill()
    filled = filled.fillna(train_means)
    return filled


def prepare_flow(sensor_ids):
    pieces = []
    raw_missing = 0
    raw_total = 0
    for year in config.years:
        frame = load_year_flow(year, sensor_ids)
        raw_missing = raw_missing + int(frame.isna().sum().sum())
        raw_total = raw_total + frame.size
        pieces.append(resample_to_15_minutes(frame))
        del frame
    combined = pd.concat(pieces)
    missing_per_sensor = combined.isna().sum(axis = 0)
    print("Raw 5 minute missing values:", raw_missing, "of", raw_total, "(" + str(round(100 * raw_missing / raw_total, 3)) + " percent)")
    print("Missing 15 minute cells after resampling:", int(missing_per_sensor.sum()), "of", combined.size)
    print("Sensors with the most missing cells:")
    print(missing_per_sensor.sort_values(ascending = False).head(5).to_string())
    filled = fill_missing_values(combined)
    print("Missing cells after filling:", int(filled.isna().sum().sum()))
    return filled


def save_cache(filled, adjacency, metadata):
    np.save(os.path.join(config.cache_directory, "flow_15min.npy"), filled.values.astype("float32"))
    np.save(os.path.join(config.cache_directory, "timestamps.npy"), filled.index.values)
    np.save(os.path.join(config.cache_directory, "adjacency.npy"), adjacency)
    metadata.to_csv(os.path.join(config.cache_directory, "sensor_metadata.csv"), index = False)


def load_cache():
    flow = np.load(os.path.join(config.cache_directory, "flow_15min.npy"))
    timestamps = pd.DatetimeIndex(np.load(os.path.join(config.cache_directory, "timestamps.npy")))
    adjacency = np.load(os.path.join(config.cache_directory, "adjacency.npy"))
    metadata = pd.read_csv(os.path.join(config.cache_directory, "sensor_metadata.csv"))
    return flow, timestamps, adjacency, metadata


def split_and_normalise(flow, timestamps, device):
    years = timestamps.year.values
    months = timestamps.month.values
    masks = {
        "train": years == 2020,
        "validation": (years == 2021) & (months <= 3),
        "test": (years == 2021) & (months >= 4),
    }
    segments = {}
    for name, mask in masks.items():
        indices = np.where(mask)[0]
        if len(indices) != indices[-1] - indices[0] + 1:
            raise ValueError("Split " + name + " is not contiguous")
        segments[name] = flow[indices[0]:indices[-1] + 1]
    # Statistics come from the training split only
    mean = float(segments["train"].mean())
    std = float(segments["train"].std())
    normalised = {}
    for name, segment in segments.items():
        normalised[name] = torch.tensor((segment - mean) / std, dtype = torch.float32, device = device)
    print("Rows per split:", {name: segment.shape[0] for name, segment in segments.items()})
    print("Train mean and std:", mean, std)
    return normalised, mean, std


def count_windows(segment):
    return segment.shape[0] - (config.input_length + config.output_length) + 1


def make_batch(segment, start_indices):
    # Windows are cut inside one split segment so no window crosses a split boundary
    offsets = torch.arange(config.input_length + config.output_length, device = segment.device)
    indices = start_indices.unsqueeze(1) + offsets.unsqueeze(0)
    window = segment[indices]
    return window[:, :config.input_length], window[:, config.input_length:]