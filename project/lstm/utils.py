import numpy as np

def filter_short_tracks(data_list, min_track_point):
    # Filtra le tracce con meno punti di min_track_point
    return [track for track in data_list if len(track['position']) >= min_track_point]

def normalize_time(data_list):
    # Normalizza il tempo in modo che ogni traccia inizi da zero
    for track in data_list:
        initial_time = track['position'][0]['time']
        for pos in track['position']:
            pos['time'] -= initial_time

    max_time = max([pos['time'] for pos in [track['position'][len(track['position']) - 1] for track in data_list]])

    for track in data_list:
        for pos in track['position']:
            pos['time'] /= max_time

def convert_to_center_coordinates(data_list):
    for track in data_list:
        for pos in track['position']:
            # Calcola il centro del rettangolo
            cx = pos['x'] + pos['w'] / 2
            cy = pos['y'] + pos['h'] / 2
            # Aggiorna il punto con le coordinate del centro
            pos['x'] = cx
            pos['y'] = cy
            # Rimuovi le chiavi 'w' e 'h' se non sono più necessarie
            del pos['w'], pos['h']

def find_90th_percentile_cadence(data_list):
    cadences = []
    for track in data_list:
        timestamps = [pos['time'] for pos in track['position']]
        # Calcola le cadenze per questa traccia e le aggiunge alla lista globale
        track_cadences = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
        cadences.extend(track_cadences)

    # Calcola il 90° percentile delle cadenze
    percentile_90_cadence = np.percentile(cadences, 90) if cadences else 0
    return percentile_90_cadence

def fill_time_gaps(data_list, min_cadence):
    for track in data_list:
        new_positions = []
        for i in range(len(track['position']) - 1):
            current_pos = track['position'][i]
            next_pos = track['position'][i + 1]
            new_positions.append(current_pos)
            time_diff = next_pos['time'] - current_pos['time']

            # Se c'è un buco temporale, riempi con valori segnaposto
            if time_diff > min_cadence:
                num_placeholders = int(time_diff / min_cadence) - 1
                for _ in range(num_placeholders):
                    placeholder = {'x': np.nan, 'y': np.nan, 'time': current_pos['time'] + min_cadence}
                    new_positions.append(placeholder)
                    current_pos['time'] += min_cadence

        # Aggiungi l'ultima posizione
        new_positions.append(track['position'][-1])
        track['position'] = new_positions

def filter_tracks_by_time_gap(data_list, percentile=90):
    # Calcola il 90° percentile di tutte le cadenze
    all_cadences = []
    for track in data_list:
        timestamps = [pos['time'] for pos in track['position']]
        track_cadences = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
        all_cadences.extend(track_cadences)

    # Se non ci sono cadenze, restituisci la lista originale
    if not all_cadences:
        return data_list

    cadence_threshold = np.percentile(all_cadences, percentile)

    # Filtra le tracce che hanno buchi temporali maggiori del 90° percentile
    filtered_data_list = []
    for track in data_list:
        timestamps = [pos['time'] for pos in track['position']]
        track_cadences = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
        if all(cadence <= cadence_threshold for cadence in track_cadences):
            filtered_data_list.append(track)

    return filtered_data_list

def pad_tracks_to_double_max_length(data_list, placeholder):
    # Trova la lunghezza massima tra tutte le tracce
    max_length = max(len(track['position']) for track in data_list)

    # Imposta la lunghezza target al doppio della lunghezza massima
    target_length = 2 * max_length
    target_length = 20

    # Effettua il padding delle tracce
    for track in data_list:
        current_length = len(track['position'])
        padding_length = target_length - current_length
        # Aggiungi elementi segnaposto alla fine di ogni traccia
        track['position'].extend([placeholder] * padding_length)

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

def sovracampiona_dati(X_train):
    nuove_tracce = []
    for traccia in X_train:
        lunghezza_traccia = len(traccia['position'])
        for i in range(lunghezza_traccia):
            # Crea una nuova traccia omettendo l'elemento in posizione i
            nuova_traccia = traccia['position'][:i] + traccia['position'][i + 1:]
            nuovo_elemento = {'position': nuova_traccia, 'id': traccia['id']}
            nuove_tracce.append(nuovo_elemento)

    # Aggiungi le nuove tracce al dataset originale
    X_train_sovracampionato = X_train + nuove_tracce
    return X_train_sovracampionato