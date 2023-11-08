import cv2
import json

def draw_tracks_on_image(image_path, track_data):
    # Carica l'immagine di sfondo
    background_image = cv2.imread(image_path)

    # Controlla se l'immagine è stata caricata correttamente
    if background_image is None:
        print("Errore nel caricamento dell'immagine.")
        return

    # Dimensioni dell'immagine
    height, width, _ = background_image.shape

    # Estrai le posizioni dalla traccia
    positions = track_data['position']

    # Inizializza il tempo relativo
    relative_time = 0
    prev_time = positions[0]['time']

    # Disegna ogni posizione sulla mappa
    for pos in positions:
        # Calcola l'incremento del tempo
        time_diff = pos['time'] - prev_time
        relative_time += time_diff
        prev_time = pos['time']

        # Converti le coordinate normalizzate in coordinate effettive (pixel)
        x = int(pos['x'] * width)
        y = int(pos['y'] * height)
        w = int(pos['w'] * width)
        h = int(pos['h'] * height)

        # Disegna il rettangolo
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        cv2.rectangle(background_image, top_left, bottom_right, (0, 255, 0), 2)

        # Disegna il tempo relativo
        cv2.putText(background_image, str(relative_time), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    1)

    # Mostra l'immagine con le tracce sovrapposte
    cv2.imshow('Tracked Image', background_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_center_and_segments_on_image(image_path, track_data):
    # Carica l'immagine di sfondo
    background_image = cv2.imread(image_path)

    # Controlla se l'immagine è stata caricata correttamente
    if background_image is None:
        print("Errore nel caricamento dell'immagine.")
        return

    # Dimensioni dell'immagine
    height, width, _ = background_image.shape

    # Estrai le posizioni dalla traccia
    positions = track_data['position']

    # Inizializza il tempo relativo e il punto precedente
    relative_time = 0
    prev_time = positions[0]['time']
    prev_point = None

    # Disegna ogni posizione sulla mappa
    for pos in positions:
        # Calcola l'incremento del tempo
        time_diff = pos['time'] - prev_time
        relative_time += time_diff
        prev_time = pos['time']

        # Converti le coordinate normalizzate in coordinate effettive (pixel)
        x = int(pos['x'] * width)
        y = int(pos['y'] * height)
        w = int(pos['w'] * width)
        h = int(pos['h'] * height)

        # Calcola il punto centrale del rettangolo
        center_x = x + w // 2
        center_y = y + h // 2
        center_point = (center_x, center_y)

        # Disegna il punto centrale
        cv2.circle(background_image, center_point, 3, (0, 0, 255), -1)

        # Disegna il segmento tra il punto centrale corrente e quello precedente
        if prev_point is not None:
            cv2.line(background_image, prev_point, center_point, (255, 0, 0), 2)

        # Aggiorna il punto precedente
        prev_point = center_point

        # Disegna il tempo relativo
        cv2.putText(background_image, str(relative_time), (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

    # Mostra l'immagine con i punti centrali e i segmenti
    cv2.imshow('Tracked Image with Centers and Segments', background_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Esempio di utilizzo
# Sostituisci 'background.jpg' con il tuo percorso dell'immagine
# e 'track_data' con i tuoi dati di traccia
image_path = 'background2.png'

with open('track.json', 'r') as f:
    track_data = json.load(f)
draw_center_and_segments_on_image(image_path, track_data)
