

import cv2                                      
import mediapipe as mp                         
from mediapipe.tasks import python              
from mediapipe.tasks.python import vision        
import csv                                      
import os                                        
import time  
import numpy as np                                    



#Hand_landmarker is mediapipes model for hand tracking.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,  # Only track one hand at a time
    min_hand_detection_confidence=0.5,  # Minimum confidence for hand detection
    min_tracking_confidence=0.5  # Minimum confidence for hand tracking
)

detector = vision.HandLandmarker.create_from_options(options)  # Create the hand landmark detector with the specified options

def normalize_landmarks(landmarks):
    """
    Normalize the hand landmarks by translating and scaling them.
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    wrist = coords[0]           # landmark 0 is always the wrist
    coords = coords - wrist     # translate: wrist becomes the origin

    scale = np.linalg.norm(coords[9])  # landmark 9 = middle finger base knuckle, use it to scale the hand size
    if scale > 0:
        coords = coords / scale

    return coords.flatten()

#setup csv file for data collection
CSV_FILE = 'training_data.csv'
file_exists = os.path.isfile(CSV_FILE)  # Check if the CSV file already exists
csv_file = open(CSV_FILE, 'a', newline='')  # Open the CSV file in append mode
csv_writer = csv.writer(csv_file)  # Create a CSV writer object, removes the need to manually handle commas and newlines

#write header row if the file is new
if not file_exists:
    header = ['label']
    for i in range(21):                
        header += [f'x{i}', f'y{i}', f'z{i}']   
    csv_writer.writerow(header)

#webcam setup
cap = cv2.VideoCapture(0)  # Open the default webcam (index 0)
print("Camera opened:", cap.isOpened())
if not cap.isOpened():
    print("ERROR: Could not open webcam")
    exit()


print("Hold a hand shape, then press the letter key to save a sample.")
print("Press 'esc' to quit.")


samples = {}
timeStamp = 0
COOLDOWN = 0.3 

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror effect

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame from BGR to RGB

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)  # Create a MediaPipe Image object from the RGB frame

    result = detector.detect(mp_image)  # Run the hand landmark detection on the frame

    current_landmarks = None  # Initialize the current landmarks to None
    if result.hand_landmarks:  # If hand landmarks are detected
        current_landmarks = result.hand_landmarks[0]  # Get the first detected hand's landmarks
        for lm in current_landmarks:  # Loop through each landmark
            x_px = int(lm.x * frame.shape[1])  # Convert normalized x coordinate to pixel value
            y_px = int(lm.y * frame.shape[0])  # Convert normalized y coordinate to pixel value
            cv2.circle(frame, (x_px, y_px), 4, (0, 255, 0), -1)  # Draw a green circle at the landmark position

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF  # Wait for a key press and get the lower byte of the key code
    if key == 27:  # If the 'esc' key is pressed, exit the loop
        break
    elif 97<=key<=122 and current_landmarks is not None:
        now = time.time()  # Get the current time
        if now-timeStamp > COOLDOWN:
            label = chr(key).upper()  # Convert the key code to a character (label)
            normalized = normalize_landmarks(current_landmarks)  # Normalize the current landmarks
            row = [label] + normalized.tolist()  # Create a row with the label and normalized landmarks
            csv_writer.writerow(row)  # Write the row to the CSV file
            csv_file.flush()  # Flush the CSV file buffer to ensure data is written

            samples[label] = samples.get(label, 0) + 1  # Update the sample count for the label
            print(f"Saved sample for label '{label}'. Total samples for this label: {samples[label]}")
            timeStamp = now  # Update the timestamp to the current time

cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows
csv_file.close()  # Close the CSV file
print("Data saved to", CSV_FILE)  




