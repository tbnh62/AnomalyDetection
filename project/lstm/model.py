import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


class LSTMAnomalyDetector:
    def __init__(self, input_dim, hidden_dim, sequence_length):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.model = self._build_model()

        # model.add(LSTM(100, activation='relu', input_shape=input_shape, return_sequences=True))
        # model.add(LSTM(50, activation='relu', return_sequences=False))
        # model.add(RepeatVector(prediction_time))
        # model.add(LSTM(50, activation='relu', return_sequences=True))
        # model.add(LSTM(100, activation='relu', return_sequences=True))
        # model.add(TimeDistributed(Dense(n_features)))
        # optimizer = Adam(clipvalue=1.0)  # Adjust the value as needed
        # model.compile(optimizer=optimizer, loss='mse')

    def _build_model(self):
        model = tf.keras.Sequential(
            [
                # Input layer con maschera per eventuali valori di padding
                tf.keras.layers.Masking(
                    mask_value=-1, input_shape=(self.sequence_length, self.input_dim)
                ),
                tf.keras.layers.LSTM(200, activation="relu", return_sequences=True),
                tf.keras.layers.Dropout(0.2),  # Dropout
                tf.keras.layers.LSTM(100, activation="relu", return_sequences=True),
                tf.keras.layers.BatchNormalization(),  # Normalizzazione
                tf.keras.layers.LSTM(
                    50,
                    activation="relu",
                    return_sequences=True,
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                ),
                # Regolarizzazione L1 ed L2
                tf.keras.layers.Dropout(0.2),  # Dropout
                tf.keras.layers.LSTM(25, activation="relu", return_sequences=False),
                tf.keras.layers.RepeatVector(
                    self.sequence_length
                ),  # Collo di bottiglia
                tf.keras.layers.LSTM(25, activation="relu", return_sequences=True),
                tf.keras.layers.Dropout(0.2),  # Dropout
                tf.keras.layers.BatchNormalization(),  # Normalizzazione
                tf.keras.layers.LSTM(50, activation="relu", return_sequences=True),
                tf.keras.layers.LSTM(100, activation="relu", return_sequences=True),
                tf.keras.layers.Dropout(0.2),  # Dropout
                tf.keras.layers.LSTM(200, activation="relu", return_sequences=True),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.input_dim)),
            ]
        )

        # Load the initializing weights
        model.load_weights("weights_epoch-91fo.h5")
        # Compila il modello con l'ottimizzatore Adam e la loss MSE
        model.compile(optimizer=Adam(clipvalue=1.0), loss="mse")

        return model

    def _build_model2(self):
        model = tf.keras.Sequential(
            [
                # Input layer con maschera per eventuali valori di padding
                tf.keras.layers.Masking(
                    mask_value=-1, input_shape=(self.sequence_length, self.input_dim)
                ),
                # Primo LSTM layer con regolarizzazione L1 ed L2
                tf.keras.layers.LSTM(
                    self.hidden_dim,
                    return_sequences=True,
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                ),
                # Dropout dopo il primo layer LSTM
                tf.keras.layers.Dropout(0.1),
                # Secondo LSTM layer
                tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True),
                # Dropout dopo il secondo layer LSTM
                tf.keras.layers.Dropout(0.1),
                # Output layer che ricostruisce l'input originale
                tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.input_dim)),
            ]
        )

        # Compila il modello con l'ottimizzatore Adam e la loss MSE
        model.compile(optimizer="adam", loss="mse")

        return model

    def train(self, train_data, epochs, batch_size):
        # Configura la callback per il salvataggio dei pesi
        checkpoint = ModelCheckpoint(
            "weights_epoch-{epoch:02d}.h5",  # il nome del file include il numero dell'epoca
            monitor="loss",  # potresti voler monitorare 'val_loss' se hai dati di validazione
            verbose=1,  # mostra messaggi dettagliati
            save_best_only=True,  # salva sempre, indipendentemente dalle performance
            save_weights_only=True,  # salva solo i pesi, non l'intero modello
            mode="auto",  # determina automaticamente il modo ('min' o 'max')
            save_freq="epoch",  # salva ad ogni epoca
        )

        # Addestra il modello usando i dati di addestramento sia come input che come target
        self.model.fit(
            train_data,
            train_data,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            callbacks=[checkpoint],  # Aggiungi la callback al processo di addestramento
        )

    def evaluate(self, test_data):
        # Predici i dati di test e calcola il Mean Squared Error (MSE)
        predicted = self.model.predict(test_data)
        mse_reference = tf.keras.losses.mean_squared_error(
            test_data.flatten(), predicted.flatten()
        )
        mse_reference_value = (
            mse_reference.numpy().mean()
        )  # Converti il tensore MSE in un valore scalare
        return mse_reference_value  #

    def evaluate2(self, test_data):
        # Predici i dati di test e calcola il Mean Squared Error (MSE)
        predicted = self.model.predict(test_data)
        mse = tf.keras.losses.mean_squared_error(test_data, predicted)
        return mse.numpy()  # Converte il risultato in un numpy array per l'analisi

    def init(self, model):
        self.model.load_weights(model)

    def get_model(self):
        return self.model
