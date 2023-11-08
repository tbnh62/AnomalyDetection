import cv2
import json
import numpy as np

import cv2
import json
import numpy as np


def draw_multiple_tracks_on_image_gray2(image_path, all_track_data):
    # Carica l'immagine di sfondo
    background_image = cv2.imread(image_path)

    # Controlla se l'immagine è stata caricata correttamente
    if background_image is None:
        print("Errore nel caricamento dell'immagine.")
        return

    # Crea un'immagine trasparente della stessa dimensione dell'immagine originale
    overlay = np.zeros_like(background_image)

    # Dimensioni dell'immagine
    height, width, _ = background_image.shape

    # Itera attraverso ogni traccia
    for track_idx, track_data in enumerate(all_track_data):
        # Estrai le posizioni dalla traccia
        positions = track_data['position']

        # Inizializza il punto precedente
        prev_point = None

        # Calcola le posizioni iniziali e finali
        x_start = positions[0]['x']
        y_start = positions[0]['y']
        x_end = positions[-1]['x']
        y_end = positions[-1]['y']

        # Scegli il colore in base alla direzione
        if y_end > y_start:
            if x_end > x_start:
                color = (30, 30, 30)  # Giù a destra
            else:
                color = (60, 60, 60)  # Giù a sinistra
        else:
            if x_end > x_start:
                color = (120, 120, 120)  # Su a destra
            else:
                color = (180, 180, 180)  # Su a sinistra

        # Disegna ogni posizione sulla mappa
        for pos in positions:
            # Converti le coordinate normalizzate in coordinate effettive (pixel)
            x = int(pos['x'] * width)
            y = int(pos['y'] * height)
            w = int(pos['w'] * width)
            h = int(pos['h'] * height)

            # Calcola il punto centrale del rettangolo
            center_x = x + w // 2
            center_y = y + h // 2
            center_point = (center_x, center_y)

            # Disegna il segmento tra il punto centrale corrente e quello precedente sull'overlay
            if prev_point is not None:
                cv2.line(overlay, prev_point, center_point, color, 2)

            # Aggiorna il punto precedente
            prev_point = center_point

    # Combina l'immagine originale e l'overlay con la trasparenza
    cv2.addWeighted(overlay, 0.5, background_image, 1.0, 0, background_image)

    # Mostra l'immagine con i segmenti di traccia
    cv2.imshow('Tracked Image with Multiple Tracks', background_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_multiple_tracks_on_image_gray(image_path, all_track_data):
    # Carica l'immagine di sfondo
    background_image = cv2.imread(image_path)

    # Controlla se l'immagine è stata caricata correttamente
    if background_image is None:
        print("Errore nel caricamento dell'immagine.")
        return

    # Dimensioni dell'immagine
    height, width, _ = background_image.shape

    # Itera attraverso ogni traccia
    for track_idx, track_data in enumerate(all_track_data):
        # Estrai le posizioni dalla traccia
        positions = track_data['position']

        # Inizializza il punto precedente
        prev_point = None

        # Calcola le posizioni iniziali e finali
        x_start = positions[0]['x']
        y_start = positions[0]['y']
        x_end = positions[-1]['x']
        y_end = positions[-1]['y']

        # Scegli il colore in base alla direzione
        if y_end > y_start:
            if x_end > x_start:
                color = (30, 30, 30)  # Giù a destra
            else:
                color = (60, 60, 60)  # Giù a sinistra
        else:
            if x_end > x_start:
                color = (120, 120, 120)  # Su a destra
            else:
                color = (180, 180, 180)  # Su a sinistra

        # Disegna ogni posizione sulla mappa
        for pos in positions:
            # Converti le coordinate normalizzate in coordinate effettive (pixel)
            x = int(pos['x'] * width)
            y = int(pos['y'] * height)
            w = int(pos['w'] * width)
            h = int(pos['h'] * height)

            # Calcola il punto centrale del rettangolo
            center_x = x + w // 2
            center_y = y + h // 2
            center_point = (center_x, center_y)

            # Disegna il segmento tra il punto centrale corrente e quello precedente
            if prev_point is not None:
                cv2.line(background_image, prev_point, center_point, color, 2)

            # Aggiorna il punto precedente
            prev_point = center_point

    # Mostra l'immagine con i segmenti di traccia
    cv2.imshow('Tracked Image with Multiple Tracks', background_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_multiple_tracks_on_image(image_path, all_track_data):
    # Carica l'immagine di sfondo
    background_image = cv2.imread(image_path)

    # Controlla se l'immagine è stata caricata correttamente
    if background_image is None:
        print("Errore nel caricamento dell'immagine.")
        return

    # Dimensioni dell'immagine
    height, width, _ = background_image.shape

    # Genera colori casuali per ciascuna traccia
    num_tracks = len(all_track_data)
    colors = np.random.randint(0, 255, size=(num_tracks, 3))

    # Itera attraverso ogni traccia
    for track_idx, track_data in enumerate(all_track_data):
        # Estrai le posizioni dalla traccia
        positions = track_data['position']

        # Inizializza il punto precedente
        prev_point = None

        # Colori per questa traccia
        color = tuple(map(int, colors[track_idx]))

        # Disegna ogni posizione sulla mappa
        for pos in positions:
            # Converti le coordinate normalizzate in coordinate effettive (pixel)
            x = int(pos['x'] * width)
            y = int(pos['y'] * height)
            w = int(pos['w'] * width)
            h = int(pos['h'] * height)

            # Calcola il punto centrale del rettangolo
            center_x = x + w // 2
            center_y = y + h // 2
            center_point = (center_x, center_y)

            # Disegna il segmento tra il punto centrale corrente e quello precedente
            if prev_point is not None:
                cv2.line(background_image, prev_point, center_point, color, 2)

            # Aggiorna il punto precedente
            prev_point = center_point

    # Mostra l'immagine con i segmenti di traccia
    cv2.imshow('Tracked Image with Multiple Tracks', background_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Esempio di utilizzo
# Sostituisci 'background.jpg' con il tuo percorso dell'immagine
# e 'all_track_data' con i tuoi dati di traccia
image_path = 'background2.png'
with open('tracks.json', 'r') as f:
    all_track_data = json.load(f)  # Supponiamo che sia un vettore di tracce
draw_multiple_tracks_on_image_gray2(image_path, all_track_data)
