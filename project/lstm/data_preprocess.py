from data_access import DataAccess
from config import Config
import numpy as np
import json

from utils import *
from local import *


class DataPreparation:
    def __init__(self) -> None:
        pass

    def preprocess(self, data_list, config, sequence_length=30):
        config = Config(f"{local_repository}\\config\\config.ini")

        sequence_length = 30  # max(len(track['position']) for track in data_list)  # Lunghezza della sequenza dopo il padding
        min_track_point = config.get_int("Data", "min_track_point")

        # Preprocessing
        data_list = filter_out_short_tracks(data_list, min_track_point)
        data_list = traslate_time(data_list)
        data_list = filter_tracks_by_time_gap(data_list)
        convert_to_center_coordinates(data_list)
        with open(
            "project\\lstm\\preprocessed tracks after center conversion.json", "w"
        ) as file:
            json.dump(data_list, file, indent=4)
        data_list = extract_and_pad_tracks(data_list, sequence_length=sequence_length)
        with open(
            "project\\lstm\\preprocessed tracks after extract_and_pad.json", "w"
        ) as file:
            json.dump(data_list, file, indent=4)
        # data_list = filter_tracks_by_time_gap(data_list)
        # data_list = filter_out_short_tracks(data_list, min_track_point=4)
        data_list = rescale_times(data_list, track_length=sequence_length)
        with open(
            "project\\lstm\\preprocessed tracks after time rescaling.json", "w"
        ) as file:
            json.dump(data_list, file, indent=4)
        # outlier filtering
        data_list = filter_outliers(data_list, sequence_length=sequence_length)
        data_list = filter_tracks_by_min_points(
            data_list, sequence_length=sequence_length, min_points=4
        )
        with open("project\\lstm\\preprocessed tracks.json", "w") as file:
            json.dump(data_list, file, indent=4)
        print("Cardinality of preprocessed data:", len(data_list))  # 22939
        padded_sequences = [
            np.array(
                [
                    [pos["x"], pos["y"], pos["w"], pos["h"], pos["time"]]
                    for pos in track["position"]
                ]
            )
            for track in data_list
        ]

        return padded_sequences
