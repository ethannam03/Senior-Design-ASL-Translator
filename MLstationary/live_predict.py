import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import joblib

# --- Load your trained model ---
model = joblib.load('trained_model.pkl')

# --- Setup MediaPipe (same as before) ---
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)


def normalize_landmarks(landmarks):
    """Same normalization used during data collection """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = coords[0]
    coords = coords - wrist
    scale = np.linalg.norm(coords[9])
    if scale > 0:
        coords = coords / scale
    return coords.flatten()


# --- Webcam setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam")
    exit()

print("Press 'Esc' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    predicted_letter = "..."
    confidence = 0.0

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]

        # draw dots for visual feedback
        for lm in hand_landmarks:
            x_px = int(lm.x * frame.shape[1])
            y_px = int(lm.y * frame.shape[0])
            cv2.circle(frame, (x_px, y_px), 4, (0, 255, 0), -1)

        # normalize the SAME way as during data collection
        features = normalize_landmarks(hand_landmarks)

        # model.predict expects a 2D array (a list of samples), so wrap in [ ]
        prediction = model.predict([features])[0]

        # predict_proba gives a probability for EACH class the model knows about
        probabilities = model.predict_proba([features])[0]
        confidence = np.max(probabilities)  # highest probability = confidence in the top prediction

        predicted_letter = prediction

    # --- Draw the UI overlay ---
    # Background rectangle for readability
    cv2.rectangle(frame, (0, 0), (300, 90), (0, 0, 0), -1)

    cv2.putText(frame, f"Letter: {predicted_letter}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.putText(frame, f"Confidence: {confidence:.1%}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("ASL Live Prediction", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Esc to quit
        break

cap.release()
cv2.destroyAllWindows()
