from data_access import DataAccess
from data_preprocess import DataPreparation
from config import Config
from model import LSTMAnomalyDetector
import numpy as np
from sklearn.model_selection import train_test_split
from utils import *
import tensorflow as tf

from local import *

config = Config("config/config.ini")


def main():
    data_access = DataAccess()
    data_list = data_access.load_data(f"{local_repository}\\data\\tracks.json")

    data_prepare = DataPreparation()
    data_list = data_prepare.preprocess(data_list, config)

    # LSTM parameters
    input_dim = 5  # x, y, w, h e time
    hidden_dim = 32  # Numero di unità LSTM
    sequence_length = 30  # max(len(track['position']) for track in data_list)  # Lunghezza della sequenza dopo il padding

    model = LSTMAnomalyDetector(input_dim, hidden_dim, sequence_length)
    model.get_model().load_weights("weights_epoch-89.h5")

    # prendo una traccia di riferimento
    reference_track = data_list[8]
    print("Reference track:\n", reference_track)

    reversed_track = invert_tracks(data_list, 2)
    perturbed_track = perturb_tracks(data_list, 2)

    sampled_tracks, inverted_tracks = invert_tracks(data_list, 3)

    for reference_track, reversed_track in zip(sampled_tracks, inverted_tracks):
        # Convert tracks to numpy arrays if they aren't already
        reference_track = np.array(reference_track)
        reversed_track = np.array(reversed_track)

        # Make predictions
        predicted_reference = model.get_model().predict(reference_track[None, :, :])
        predicted_reverse = model.get_model().predict(reversed_track[None, :, :])

        # Calculate MSE
        mse_reference = tf.keras.losses.mean_squared_error(
            reference_track.flatten(), predicted_reference.flatten()
        )
        mse_reverse = tf.keras.losses.mean_squared_error(
            reversed_track.flatten(), predicted_reverse.flatten()
        )

        mse_reference_value = (
            mse_reference.numpy().mean()
        )  # Convert tensor MSE to scalar value
        mse_reverse_value = (
            mse_reverse.numpy().mean()
        )  # Convert tensor MSE to scalar value

        print("MSE for reference track:", mse_reference_value)
        print("MSE for reversed track:", mse_reverse_value)


if __name__ == "__main__":
    main()
