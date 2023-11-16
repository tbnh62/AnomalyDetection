import numpy as np
import random
import math
import copy

from sklearn.ensemble import IsolationForest


def filter_out_short_tracks(data_list, min_track_point):
    # Filtra le tracce con meno punti di min_track_point
    return [track for track in data_list if len(track["position"]) >= min_track_point]


def filter_patchy_tracks(data_list):
    filtered_tracks = []
    for track in data_list:
        points = track["position"]
        valid_trajectory = True
        for i in range(len(points) - 1):
            if points[i + 1]["time"] - points[i]["time"] > 3000:
                valid_trajectory = False
                break
        if valid_trajectory:
            filtered_tracks.append(track)

    return filtered_tracks


def find_closest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def embed_and_pad(track, min_track_length):
    # Determine the range for time instants. Maybe this step should be included in the extract_and_pad routine
    m = min_track_length
    last_time = track["position"][-1]["time"]
    time_instants = [
        math.ceil(k * 1000 / m)
        for k in range(max(m, math.ceil(last_time / (1000 / m)) + 1))
    ]

    # Create a dictionary to store points for each time instant
    points_dict = {time: [] for time in time_instants}

    # Assign points to the closest time instant
    for point in track["position"]:
        closest_time = math.ceil(find_closest(time_instants, point["time"]))
        points_dict[closest_time].append(point)

    # Calculate median values and create super track
    super_track = []
    for time in time_instants:
        if points_dict[time]:
            x = np.median([p["x"] for p in points_dict[time]])
            y = np.median([p["y"] for p in points_dict[time]])
            w = np.median([p["w"] for p in points_dict[time]])
            h = np.median([p["h"] for p in points_dict[time]])
            super_track.append(
                {
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "w": round(w, 2),
                    "h": round(h, 2),
                    "time": time,
                }
            )
        else:
            super_track.append({"x": -1, "y": -1, "w": -1, "h": -1, "time": time})

    return super_track


def extract_sub_tracks(super_track, sub_track_length):
    # Extract all possible sub-tracks of length 10
    return [
        copy.deepcopy(super_track[i : i + sub_track_length])
        for i in range(len(super_track) - sub_track_length + 1)
    ]


def extract_and_pad_tracks(tracks, sequence_length):
    new_tracks = []
    for track in tracks:
        super_track = embed_and_pad(track, min_track_length=sequence_length)
        sub_tracks = extract_sub_tracks(super_track, sub_track_length=sequence_length)
        for sub_track in sub_tracks:
            new_tracks.append({"position": sub_track, "id": track["id"]})
    return new_tracks


def traslate_time(data_list):
    # Normalizza il tempo in modo che ogni traccia inizi da zero
    for track in data_list:
        initial_time = track["position"][0]["time"]
        for pos in track["position"]:
            pos["time"] -= initial_time

    return data_list

    """
    max_time = max(
        [
            pos["time"]
            for pos in [
                track["position"][len(track["position"]) - 1] for track in data_list
            ]
        ]
    )

    for track in data_list:
        for pos in track["position"]:
            pos["time"] /= max_time
    """


def all_time_gaps(data_list):
    time_gaps = []
    for track in data_list:
        if track["position"]:
            initial_time = track["position"][0]["time"]
            for pos in track["position"]:
                time_gap = pos["time"] - initial_time
                pos["time"] = time_gap
                time_gaps.append(time_gap)
    return time_gaps


def convert_to_center_coordinates(data_list):
    for track in data_list:
        for pos in track["position"]:
            # Calcola il centro del rettangolo
            cx = pos["x"] + pos["w"] / 2
            cy = pos["y"] + pos["h"] / 2
            # Aggiorna il punto con le coordinate del centro
            pos["x"] = cx
            pos["y"] = cy
            # Rimuovi le chiavi 'w' e 'h' se non sono più necessarie
            # del pos["w"], pos["h"]


def find_90th_percentile_cadence(data_list):
    cadences = []
    for track in data_list:
        timestamps = [pos["time"] for pos in track["position"]]
        # Calcola le cadenze per questa traccia e le aggiunge alla lista globale
        track_cadences = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
        cadences.extend(track_cadences)

    # Calcola il 90° percentile delle cadenze
    percentile_90_cadence = np.percentile(cadences, 90) if cadences else 0
    return percentile_90_cadence


def fill_time_gaps(data_list, min_cadence):
    for track in data_list:
        new_positions = []
        for i in range(len(track["position"]) - 1):
            current_pos = track["position"][i]
            next_pos = track["position"][i + 1]
            new_positions.append(current_pos)
            time_diff = next_pos["time"] - current_pos["time"]

            # Se c'è un buco temporale, riempi con valori segnaposto
            if time_diff > min_cadence:
                num_placeholders = int(time_diff / min_cadence) - 1
                for _ in range(num_placeholders):
                    placeholder = {
                        "x": np.nan,
                        "y": np.nan,
                        "time": current_pos["time"] + min_cadence,
                    }
                    new_positions.append(placeholder)
                    current_pos["time"] += min_cadence

        # Aggiungi l'ultima posizione
        new_positions.append(track["position"][-1])
        track["position"] = new_positions


def filter_tracks_by_time_gap(data_list, percentile=90):
    # Calcola il 90° percentile di tutte le cadenze
    all_cadences = []
    for track in data_list:
        timestamps = [pos["time"] for pos in track["position"]]
        track_cadences = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
        all_cadences.extend(track_cadences)

    # Se non ci sono cadenze, restituisci la lista originale
    if not all_cadences:
        return data_list

    cadence_threshold = np.percentile(all_cadences, percentile)

    # Filtra le tracce che hanno buchi temporali maggiori del percentile del 90°
    filtered_data_list = []
    for track in data_list:
        timestamps = [pos["time"] for pos in track["position"]]
        track_cadences = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
        if all(cadence <= cadence_threshold for cadence in track_cadences):
            filtered_data_list.append(track)

    return filtered_data_list


def filter_outliers(tracks, sequence_length):
    # Initialize filtered_tracks with empty dictionaries for each track
    filtered_tracks = [{"position": [], "id": track["id"]} for track in tracks]

    for i in range(sequence_length):
        # Extract the i-th point from each track and convert to a list of numerical values
        ith_points = [
            list(track["position"][i].values())
            for track in tracks
            if len(track["position"]) > i
        ]

        # Convert to NumPy array for Isolation Forest
        ith_points_array = np.array(ith_points)

        # Apply Isolation Forest to detect outliers
        clf = IsolationForest(random_state=42)
        clf.fit(ith_points_array)
        is_inlier = clf.predict(ith_points_array)

        # Append non-outlier points to the corresponding track in filtered_tracks
        for track_index, inlier in enumerate(is_inlier):
            if inlier == 1:
                filtered_tracks[track_index]["position"].append(
                    tracks[track_index]["position"][i]
                )

    # Ensure each track's 'position' has 'sequence_length' points
    for track in filtered_tracks:
        while len(track["position"]) < sequence_length:
            track["position"].append(
                {"x": -1, "y": -1, "w": -1, "h": -1, "time": -1}
            )  # Replace with your default value

    return filtered_tracks


def rescale_times(tracks, track_length):
    for track in tracks:
        # Extracting the timestamp from the ID and rounding the first time instant
        id_parts = track["id"].split("-")
        first_time = round(track["position"][0]["time"])
        id_parts[-1] = str(int(id_parts[-1]) + first_time)
        track["id"] = "-".join(id_parts)

        # Updating time instants
        for i, point in enumerate(track["position"]):
            if all(
                coord == -1
                for coord in [point["x"], point["y"], point["w"], point["h"]]
            ):
                point["time"] = -1

            else:
                # Scale time instants from 0 to 1
                point["time"] = round(i / (track_length - 1), 2)

    return tracks


def pad_tracks_to_double_max_length(data_list, placeholder):
    # Trova la lunghezza massima tra tutte le tracce
    max_length = max(len(track["position"]) for track in data_list)

    # Imposta la lunghezza target al doppio della lunghezza massima
    target_length = 2 * max_length
    # target_length = 20

    # Effettua il padding delle tracce
    for track in data_list:
        current_length = len(track["position"])
        padding_length = target_length - current_length
        # Aggiungi elementi segnaposto alla fine di ogni traccia
        track["position"].extend([placeholder] * padding_length)

    return data_list


def pad_tracks(track, placeholder):
    target_length = 20

    current_length = track.shape[0]
    padding_length = target_length - current_length
    # Crea un array di padding
    padding = np.full((padding_length, track.shape[1]), placeholder)
    # Concatena la traccia con il padding
    padded_track = np.concatenate((track, padding), axis=0)

    return padded_track


def resample_data(X_train):
    nuove_tracce = []
    for traccia in X_train:
        lunghezza_traccia = len(traccia["position"])
        for i in range(lunghezza_traccia):
            # Crea una nuova traccia omettendo l'elemento in posizione i
            nuova_traccia = traccia["position"][:i] + traccia["position"][i + 1 :]
            nuovo_elemento = {"position": nuova_traccia, "id": traccia["id"]}
            nuove_tracce.append(nuovo_elemento)

    # Aggiungi le nuove tracce al dataset originale
    X_train_sovracampionato = X_train + nuove_tracce
    return X_train_sovracampionato


def track_to_string(track):
    points_str = ", ".join(
        [
            f'{{"x": {p["x"]}, "y": {p["y"]}, "w": {p["w"]}, "h": {p["h"]}, "time": {p["time"]}}}'
            for p in track["position"]
        ]
    )
    return f'{{"position": [{points_str}], "id": {track["id"]}}}'


def print_random_sample(message, tracks, sample_size):
    print(message + ":\n")
    sample_size = min(
        sample_size, len(tracks)
    )  # Ensure sample size is not greater than the dataset size
    sample_tracks = random.sample(tracks, sample_size)

    for track in sample_tracks:
        print(track_to_string(track))


def invert_tracks(tracks, num_samples):
    # Select a random sample of tracks
    sampled_tracks = random.sample(tracks, num_samples)

    inverted_tracks = []

    for track in sampled_tracks:
        track_length = len(track)
        inverted_track = []

        # Reverse the order of the points in the track
        reversed_points = track[::-1]

        for i, point in enumerate(reversed_points):
            if all(coord == -1 for coord in point[:4]):
                # If all coordinates are -1, pad the entire point with -1
                inverted_point = [-1, -1, -1, -1, -1]
            else:
                # Keep the coordinates as they are and recalculate the time
                inverted_point = list(point[:4]) + [round(i / (track_length - 1), 2)]

            inverted_track.append(inverted_point)

        inverted_tracks.append(np.array(inverted_track))

        # Print original and inverted track
        print("Original Track:")
        print(track)
        print("Inverted Track:")
        print(np.array(inverted_track))

    return sampled_tracks, inverted_tracks


def perturb_tracks(tracks, num_samples):
    # Select a random sample of tracks
    sampled_tracks = random.sample(tracks, num_samples)

    transformed_tracks = []

    for track in sampled_tracks:
        transformed_track = []

        for point in track:
            if all(coord == -1 for coord in point[:4]):
                # If all coordinates are -1, keep the point as it is
                transformed_point = point.copy()
            else:
                # Apply transformation to the y coordinate
                x, y, w, h, time = point
                transformed_y = round(0.5 * math.sin(2 * math.pi * y), 2)
                transformed_point = [x, transformed_y, w, h, time]

            transformed_track.append(transformed_point)

        transformed_tracks.append(np.array(transformed_track))

        # Print original and perturbed track
        print("Original Track:")
        print(track)
        print("Perturbed Track:")
        print(np.array(transformed_track))

    return sampled_tracks, transformed_tracks
