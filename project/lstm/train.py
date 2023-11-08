from data_access import DataAccess
from config import Config
from model import LSTMAnomalyDetector
import numpy as np
from sklearn.model_selection import train_test_split
from utils import *

config = Config('config/config.ini')

def main():
    data_access = DataAccess()
    data_list = data_access.load_data('data/tracks.json')

    # Elimina tutte le tracce troppo piccole (numero di punti < config.min_track_point)
    min_track_point = config.get_int("Data", "min_track_point")
    data_list = filter_short_tracks(data_list, min_track_point)

    # Normalizza il tempo per ogni traccia
    normalize_time(data_list)

    # Converti le coordinate del rettangolo in coordinate centrali
    convert_to_center_coordinates(data_list)

    # Trova la cadenza minima tra tutti i punti
    ## min_cadence = find_90th_percentile_cadence(data_list)
    # Riempie i buchi temporali in ogni traccia
    ## fill_time_gaps(data_list, min_cadence)

    data_list = filter_tracks_by_time_gap(data_list)

    # Sovracampiona i dati
    data_list = sovracampiona_dati(data_list)

    # Placeholder può essere un dizionario con valori NaN o un altro valore che indica dati mancanti
    placeholder_value = {'x': -1, 'y': -1, 'time': -1}

    # Chiamata alla funzione
    padded_data_list = pad_tracks_to_double_max_length(data_list, placeholder_value)

    # Supponendo che `tracks` sia la lista delle tue tracce
    padded_sequences = [np.array([[pos['x'], pos['y'], pos['time']] for pos in track['position']]) for track in data_list]

    # Dividi i dati in set di addestramento e test
    X_train, X_test = train_test_split(padded_sequences, test_size=0.2, random_state=42)

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Inizializza il modello LSTM
    input_dim = 3  # x e y, time
    hidden_dim = 32  # Numero di unità LSTM
    sequence_length = 20  # max(len(track['position']) for track in data_list)  # Lunghezza della sequenza dopo il padding
    model = LSTMAnomalyDetector(input_dim, hidden_dim, sequence_length)

    # Addestra il modello
    epochs = 1024
    batch_size = 32

    model.train(X_train, epochs, batch_size)

    # Valuta il modello
    mse = model.evaluate(X_test)
    print(f"MSE sui dati di test: {np.mean(mse)}")

    # Fai ulteriori operazioni come l'addestramento del modello qui

if __name__ == '__main__':
    main()
