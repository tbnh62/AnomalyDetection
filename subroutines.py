import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from random import sample

from parameters import *

from dataengineering import *
from dataanalysis import *


def read():
    global traj_vt2
    with open(f"{local_repository}\\{path_vt2_dataset}", "r") as json_file:
        traj_vt2 = json.load(json_file)


def describeRawData(data):
    json_description = Json()

    # Counting total json data
    json_count = json_description.count_elements_in_json(path_vt2_dataset_office)
    # Counting negative and zero x_values
    negative_x_count = sum(1 for item in data for p in item["position"] if p["x"] < 0)
    zero_x_count = sum(1 for item in data for p in item["position"] if p["x"] == 0.0)

    return json_count, negative_x_count, zero_x_count


def plotDataDensity(data, variable="time"):
    values = [p[variable] for item in data for p in item["position"]]
    time_density_visualization = DataExploration()
    time_density_visualization.plot_density(
        all_t_values, bin_width=0.05, xlabel="Normalized timestamps"
    )


def prepareDataset(data):
    # Adjusting t_values based on initial t_value
    for item in data:
        initial_t_value = item["position"][0]["time"]
        for p in item["position"]:
            p["time"] -= initial_t_value

    # Min-max normalization for t_values
    all_t_values = [p["time"] for item in data for p in item["position"]]
    min_t = min(all_t_values)
    max_t = max(all_t_values)
    for item in data:
        for p in item["position"]:
            p["time"] = (p["time"] - min_t) / (max_t - min_t) if max_t != min_t else 0.0

    return data


read()
json_count, negative_x_count, zero_x_count = describeRawData(traj_vt2)
processed_data = prepareDataset(traj_vt2)

print(f"Cardinality: {json_count}")  # Cardinality: 5899
print(
    f"Number of negative x_values: {negative_x_count}"
)  # Number of negative x_values: 367
print(f"Number of zero x_values: {zero_x_count}")  # Number of zero x_values: 17
print("Processed Data:", sample(processed_data, 10))

plotTimeDensity(traj_vt2)
