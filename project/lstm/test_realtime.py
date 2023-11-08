import cv2
from ultralytics import YOLO
import numpy as np
from model import LSTMAnomalyDetector
from collections import defaultdict
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# Store the track history
track_history = defaultdict(lambda: [])

# Define the standard frame size (change these values as needed)
standard_width = 640
standard_height = 480

def display_text(frame, text, position):
    # Converti le coordinate in interi prima di utilizzarle
    position = (int(position[0]), int(position[1]))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 0)  # Green color
    font_thickness = 2
    cv2.putText(frame, text, position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

def prepare_sequence(track):
    # Convert track data to the input format expected by the LSTM model
    # Assuming the LSTM model takes normalized coordinates and time as input
    sequence = np.array(track, dtype=np.float32)
    # Normalize x, y coordinates
    sequence[:, :2] /= np.array([standard_width, standard_height])
    # Add time as the third dimension
    sequence = np.hstack((sequence, (600 / 69533) * np.arange(len(track)).reshape(-1, 1)))
    return sequence.reshape(1, sequence_length, input_dim)  # LSTM expects input shape (batch_size, sequence_length, input_dim)

def display_anomaly_score(frame, score, position):
    # Display the anomaly score on the video frame
    text = f"Anomaly Score: {score:.2f}"
    display_text(frame, text, position)

def display_anomaly_score2(frame, score, position):
    # Display the anomaly score on the video frame
    text = f"{score:.2f}"
    display_text(frame, text, position)

# Load the YOLO model
model = YOLO('yolov8n.pt')

# Initialize the LSTM model
input_dim = 3  # assuming x, y, and time as inputs
hidden_dim = 32
sequence_length = 20
lstm_model = LSTMAnomalyDetector(input_dim, hidden_dim, sequence_length)
lstm_model.get_model().load_weights('weights_epoch-18.h5')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize tracking data structure
tracks = {}  # Dictionary to keep track of all tracks

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (standard_width, standard_height))

    # Perform object detection
    results = model.track(frame, persist=True)

    if results[0].boxes is not None:  # Check if there are results and boxes
        # Get the boxes
        boxes = results[0].boxes.xywh.cpu()

        if results[0].boxes.id is not None:
            # If 'int' attribute exists (there are detections), get the track IDs
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Loop through the detections and add data to the DataFrame
            anomaly_text = ""  # Initialize the anomaly text
            for i, box in zip(range(0, len(track_ids)), results[0].boxes.xywhn.cpu()):
                x, y, w, h = box
        else:
            anomaly_text = ""
            # If 'int' attribute doesn't exist (no detections), set track_ids to an empty list
            track_ids = []

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        display_text(annotated_frame, anomaly_text, (10, 30))  # Display the anomaly text
        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        # Loop through each track
        for track_id, track in track_history.items():
            if len(track) >= sequence_length:
                # Use only the last sequence_length points for the LSTM model
                last_sequence = track[-sequence_length:]
                # Prepare the sequence for the LSTM model
                sequence = prepare_sequence(last_sequence)
                # Perform the anomaly detection
                anomaly_score = lstm_model.evaluate(sequence)
                # Display the anomaly score on the video frame
                display_anomaly_score2(annotated_frame, anomaly_score, last_sequence[-1][:2])

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

    else:
        # If no detections, display the original frame without annotations
        cv2.imshow("YOLOv8 Tracking [n]", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
