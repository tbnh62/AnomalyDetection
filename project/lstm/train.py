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

    data_list = extract_and_pad_tracks(data_list)
    print(len(data_list))  # 11971

    data_list = rescale_times(data_list)

    with open("preprocessed tracks.json", "w") as file:
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

    # Inizializza il modello LSTM
    input_dim = 5  # x, y, w, h e time
    hidden_dim = 32  # Numero di unità LSTM
    sequence_length = 15  # max(len(track['position']) for track in data_list)  # Lunghezza della sequenza dopo il padding
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

## sequence_length = 15
# loss: 0.3244
# loss: 0.2766
# loss: 0.2731
# loss: 0.2715
# loss: 0.2702
# loss: 0.2691
# loss
