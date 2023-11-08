from data_access import DataAccess
from config import Config
from model import LSTMAnomalyDetector
import numpy as np
from sklearn.model_selection import train_test_split
from utils import *
import tensorflow as tf

config = Config('config/config.ini')

def main():
    # Inizializza il modello LSTM
    input_dim = 3  # x e y, time
    hidden_dim = 32  # Numero di unità LSTM
    sequence_length = 20 # max(len(track['position']) for track in data_list)  # Lunghezza della sequenza dopo il padding
    model = LSTMAnomalyDetector(input_dim, hidden_dim, sequence_length)

    model.get_model().load_weights('weights_epoch-18.h5')

    data_access = DataAccess()
    data_list = data_access.load_data('data/tracks.json')

    # Elimina tutte le tracce troppo piccole (numero di punti < config.min_track_point)
    min_track_point = config.get_int("Data", "min_track_point")
    data_list = filter_short_tracks(data_list, min_track_point)

    normalize_time(data_list)
    convert_to_center_coordinates(data_list)
    data_list = filter_tracks_by_time_gap(data_list)
    data_list = [np.array([[pos['x'], pos['y'], pos['time']] for pos in track['position']]) for track in data_list]

    # prendo una traccia di riferimento
    reference_track = data_list[5]

    # Supponiamo che tu abbia i seguenti array
    x = reference_track[:, 0]  # il tuo array x
    y = reference_track[:, 1]  # il tuo array y
    time = reference_track[:, 2]  # il tuo array time

    # Inverti x e y
    x_invertito = x[::-1]
    y_invertito = y[::-1]

    # Espandi le dimensioni degli array per farli diventare 2D (con dimensione aggiuntiva come colonna)
    x_invertito_2d = x_invertito[:, np.newaxis]
    y_invertito_2d = y_invertito[:, np.newaxis]
    time_2d = time[:, np.newaxis]

    # Concatena lungo l'asse delle colonne per creare un singolo array 2D
    reversed_track = np.concatenate([x_invertito_2d, y_invertito_2d, time_2d], axis=1)

    print(reversed_track)

    reference_track = pad_tracks(reference_track, [-1, -1, -1])
    reversed_track = pad_tracks(reversed_track, [-1, -1, -1])

    # Ora passa la traccia con posizioni invertite al modello.
    predicted_reference = model.get_model().predict(reference_track[None, :, :])
    predicted_reverse = model.get_model().predict(reversed_track[None, :, :])

    # Calcola l'MSE
    mse_reference = tf.keras.losses.mean_squared_error(reference_track.flatten(), predicted_reference.flatten())
    mse_reverse = tf.keras.losses.mean_squared_error(reference_track.flatten(), predicted_reverse.flatten())

    mse_reference_value = mse_reference.numpy().mean()  # Converti il tensore MSE in un valore scalare
    mse_reverse_value = mse_reverse.numpy().mean()  # Converti il tensore MSE in un valore scalare

    print("MSE per la traccia di riferimento:", mse_reference_value)
    print("MSE per la traccia con posizioni invertite:", mse_reverse_value)

if __name__ == '__main__':
    main()
