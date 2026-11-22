# AI651: PA3 (25280058)

Spatio-temporal traffic forecasting on LargeST San Diego (716 sensors, 15 minute flow, 8 steps in and 4 steps out), comparing temporal-only, fixed, adaptive and hybrid adjacency.

Run every command from inside this folder. Random seed: 2026.

## Files
- config.py: paths, seed, hyperparameters and the skip connection switch
- common/: data_utils.py, graph_utils.py, models.py, train_utils.py
- q1_aggregation.py: Q1 checks (numpy only)
- q2_ring_road.py: Q2 checks and heatmaps (numpy and matplotlib only)
- q3_data_graph.py: Q3 resampling, graph statistics, caching
- q4_q5_train.py: trains temporal-only, fixed, adaptive and hybrid models
- q6_analysis.py: final table, prediction plots, learned graph analysis
- results/ and figures/: outputs of the main run
- ablations/: outputs of the first run without the skip connection

## Main configuration (the results in results/ and figures/)
- use_skip_connection = True, so the graph layer is H = ReLU(A Z W) + Z
- number_of_epochs = 40, patience 8, batch size 64, learning rate 1e-3, L1 loss
- hidden dimension 32, embedding dimension 10, top k = 20 (applied before the softmax), learnable alpha starting at 0.5
- Hardware: Tesla T4 (Kaggle), 4 CPU cores, torch 2.11

## Ablation (the results in ablations/)
The first run used the literal layer H = ReLU(A Z W) with no skip connection and 30 epochs. To reproduce it, edit config.py:
```
use_skip_connection = False
number_of_epochs = 30
```
and run the pipeline again from step 4 below. Everything else is identical.

## Running on Kaggle (T4 GPU)
1. Create a Kaggle notebook, set the accelerator to GPU T4 and turn internet on if you need pip.
2. Add two datasets to the notebook:
   - the public LargeST dataset: https://www.kaggle.com/datasets/liuxu77/largest
   - this code folder zipped and uploaded as a dataset
3. In the notebook run:
```
!cp -r /kaggle/input/<your-code-dataset-name>/25280058_PA3 /kaggle/working/
%cd /kaggle/working/25280058_PA3
!pip install -q tables
```
4. Run the pipeline in order:
```
!python q1_aggregation.py
!python q2_ring_road.py
!python q3_data_graph.py
!python q4_q5_train.py
!python q6_analysis.py
```
5. Optional smoke test of training before the full run:
```
!python q4_q5_train.py --models temporal,fixed --epochs 2
```
6. Zip the outputs for download (cache is skipped because it can be rebuilt):
```
import os, zipfile
output_root = "/kaggle/working/outputs"
zip_path = "/kaggle/working/pa3_outputs.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for folder in ["figures", "results"]:
        for directory, _, file_names in os.walk(os.path.join(output_root, folder)):
            for file_name in file_names:
                full_path = os.path.join(directory, file_name)
                archive.write(full_path, os.path.relpath(full_path, output_root))
```

## Running locally
Put ca_meta.csv, ca_rn_adj.npy and ca_his_raw_2020.h5 and ca_his_raw_2021.h5 from LargeST under data_raw/ and run the same five commands. A GPU is optional but a CPU run is slow. Q3 needs a few GB of RAM because it reads the full California files.

## Notes
- San Diego sensors are selected as District 11 in ca_meta.csv. The script prints the count and warns if it is not 716.
- Flow is a count, so three 5 minute steps are summed to one 15 minute step. Missing values are forward filled, and leading gaps use the 2020 mean of the sensor.
- Normalisation statistics come from 2020 only. Windows never cross a split boundary.
- The road adjacency is not symmetric, so the fixed graph uses the row normalisation D^-1 (A + I). The hybrid graph also uses the row form.
- Timings include the validation pass after every epoch and come from a single seed.
- Outputs are written to outputs/ (figures, results, cache). Do not commit outputs/cache or data_raw/.