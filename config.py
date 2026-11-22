import os

on_kaggle = os.path.exists("/kaggle/working")
raw_data_root = "/kaggle/input" if on_kaggle else "data_raw"
output_directory = "/kaggle/working/outputs" if on_kaggle else "outputs"
cache_directory = os.path.join(output_directory, "cache")
figure_directory = os.path.join(output_directory, "figures")
result_directory = os.path.join(output_directory, "results")

seed = 2026
san_diego_district = 11 # San Diego sensors are the District 11 rows of the CA metadata
expected_number_of_nodes = 716
years = [2020, 2021]

input_length = 8
output_length = 4
hidden_dimension = 32
embedding_dimension = 10
top_k = 20
learn_alpha = True # If False, alpha stays fixed at 0.5 in the hybrid graph
use_skip_connection = True # Adds the temporal features back after the graph layer, set False for the literal ReLU(A Z W) layer

batch_size = 64
evaluation_batch_size = 256
learning_rate = 1e-3
number_of_epochs = 40
patience = 8

plot_sensor_indices = [50, 300, 600]
plot_week_start = "2020-03-02"
plot_horizon_index = 3 # 0 is the 15 minute horizon and 3 is the 60 minute horizon


def make_directories():
    for directory in [output_directory, cache_directory, figure_directory, result_directory]:
        os.makedirs(directory, exist_ok = True)