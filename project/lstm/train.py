from data_access import DataAccess
from config import Config
from model import LSTMAnomalyDetector
import numpy as np
import json

from sklearn.model_selection import train_test_split
from utils import *
from local import *

config = Config(f"{local_repository}\\config\\config.ini")


def main():
    data_access = DataAccess()
    data_list = data_access.load_data(f"{local_repository}\\data\\tracks.json")

    # LSTM parameters
    input_dim = 5  # x, y, w, h e time
    hidden_dim = 32  # Numero di unità LSTM
    sequence_length = 30  # max(len(track['position']) for track in data_list)  # Lunghezza della sequenza dopo il padding

    # Elimina tutte le tracce troppo piccole (numero di punti < config.min_track_point)
    min_track_point = config.get_int("Data", "min_track_point")
    data_list = filter_short_tracks(data_list, min_track_point)
    # print_random_sample("After filtering for short tracks", data_list, 5)

    # Normalizza il tempo per ogni traccia
    normalize_time(data_list)
    """
    print_random_sample(
        "After subtracting the initial time to all the time instants", data_list, 5
    )
    """

    # Converti le coordinate del rettangolo in coordinate centrali
    convert_to_center_coordinates(data_list)
    # print_random_sample("After converting to center coordinates", data_list, 5)

    # Trova la cadenza minima tra tutti i punti
    ## min_cadence = find_90th_percentile_cadence(data_list)
    # Riempie i buchi temporali in ogni traccia
    ## fill_time_gaps(data_list, min_cadence)

    data_list = filter_tracks_by_time_gap(data_list)

    data_list = extract_and_pad_tracks(data_list, sequence_length=sequence_length)
    print(len(data_list))  # 22939

    data_list = rescale_times(data_list, track_length=sequence_length)

    with open("project\\lstm\\preprocessed tracks.json", "w") as file:
        json.dump(data_list, file, indent=4)

    # print_random_sample("After extracting and padding", data_list, 5)

    # Sovracampiona i dati
    # data_list = resample_data(data_list)

    # Placeholder può essere un dizionario con valori NaN o un altro valore che indica dati mancanti
    placeholder_value = {"x": -1, "y": -1, "w": -1, "h": -1, "time": -1}

    # Chiamata alla funzione
    # padded_data_list = pad_tracks_to_double_max_length(data_list, placeholder_value)

    # Supponendo che `tracks` sia la lista delle tue tracce
    padded_sequences = [
        np.array(
            [
                [pos["x"], pos["y"], pos["w"], pos["h"], pos["time"]]
                for pos in track["position"]
            ]
        )
        for track in data_list
    ]

    # Dividi i dati in set di addestramento e test
    X_train, X_test = train_test_split(padded_sequences, test_size=0.2, random_state=42)

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    model = LSTMAnomalyDetector(input_dim, hidden_dim, sequence_length)

    # Addestra il modello
    epochs = 1024
    batch_size = 32

    model.train(X_train, epochs, batch_size)

    # Valuta il modello
    mse = model.evaluate(X_test)
    print(f"MSE sui dati di test: {np.mean(mse)}")

    # Fai ulteriori operazioni come l'addestramento del modello qui


if __name__ == "__main__":
    main()

## sequence_length = 30
# epoch 1 loss: 0.2211
# epoch 2 loss: 0.1867
# epoch 3 loss: 0.1739
# epoch 4 loss: 0.1685
# epoch 5 loss 0.1718 did not improve
# epoch 6 loss 0.1701
# epoch 7 loss 0.1660
# epoch 11 loss 0.1469
# epoch 12 loss 0.1275
# epoch 13 loss 0.1203
# epoch 14 loss 0.1061
# epoch 15 loss 0.0947
# epoch 17 loss 0.0877
# epoch 19 loss 0.1240
# epoch 20 loss: 0.1249
# epoch 21 loss: 0.1314
# epoch 22 loss: 0.0936
# epoch 23 loss: 0.0816
# epoch 26 loss: 0.0569
# epoch 27 loss: 0.1053
# epoch 28 loss: 0.0771
# epoch 29 loss: 0.0524 weights_epoch-29.h5
# epoch 30 loss: 0.0544
# epoch 31 loss: 0.0499
# epoch 32 loss: 0.0488 weights_epoch-32.h5
# epoch 34 loss: 0.0409 weights_epoch-34.h5
# epoch 36 loss: 0.0367 weights_epoch-36.h5
# epoch 39 loss: 0.0359 weights_epoch-39.h5
# epoch 40 loss: 0.0328 weights_epoch-40.h5
# epoch 42 loss: 0.0317 weights_epoch-42.h5
# epoch 44 loss: 0.0291 weights_epoch-44.h5
# epoch 47 loss: 0.0276 weights_epoch-47.h5
# epoch 51 loss: 0.0243 weights_epoch-51.h5
# epoch 55 loss: 0.0233 weights_epoch-55.h5
# epoch 56 loss: 0.0213 weights_epoch-56.h5
# epoch 63 loss: 0.0204 weights_epoch-63.h5
# epoch 65 loss: 0.0183 weights_epoch-65.h5
# epoch 69 loss: 0.0181 weights_epoch-69.h5
# epoch 73 loss: 0.0167 weights_epoch-73.h5
# epoch 82 loss: 0.0154 weights_epoch-82.h5
# epoch 86 loss: 0.0141 weights_epoch-86.h5
# epoch 89 loss: 0.0133 weights_epoch-89.h5
