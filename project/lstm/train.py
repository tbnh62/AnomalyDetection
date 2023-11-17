# coding=utf-8

from data_access import DataAccess
from data_preprocess import DataPreparation
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

    data_prepare = DataPreparation()
    padded_sequences = data_prepare.preprocess(data_list, config)

    # LSTM parameters
    input_dim = 5  # x, y, w, h e time
    hidden_dim = 32  # Numero di unità LSTM
    sequence_length = 30

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
