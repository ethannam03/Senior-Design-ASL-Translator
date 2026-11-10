import os
import sys
import urllib.request

import cv2
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# MediaPipe Hands landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_MCP = 17
PINKY_TIP = 20

FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
NUM_LANDMARKS = 21


# Resolved next to this file, so it works no matter which folder the script
# is launched from (VS Code, a desktop shortcut, autostart, ...).
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hand_landmarker.task')
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          
    (0, 5), (5, 6), (6, 7), (7, 8),          
    (5, 9), (9, 10), (10, 11), (11, 12),     
    (9, 13), (13, 14), (14, 15), (15, 16),   
    (13, 17), (17, 18), (18, 19), (19, 20),  
    (0, 17),                                 
]


def ensure_model():
    """Download the hand model next to this file if it isn't there yet."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        return
    print(f"Model not found at {MODEL_PATH}; downloading (~8 MB)...")
    tmp_path = MODEL_PATH + ".part"
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp_path)
        os.replace(tmp_path, MODEL_PATH)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        sys.exit(
            f"Could not download the model automatically ({e}).\n"
            "Download it manually with:\n"
            f"  wget -O hand_landmarker.task {MODEL_URL}"
        )


def create_hand_landmarker():
    ensure_model()
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.4,
        min_tracking_confidence=0.4,

    )
    return mp_vision.HandLandmarker.create_from_options(options)

def draw_landmarks(frame, landmarks):

    h,w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    #make circles
    for point in points:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)
    #make lines between points
    for p1,p2 in HAND_CONNECTIONS:
        cv2.line(frame, points[p1], points[p2], (0, 255, 0), 2)
    
def landmarks_to_array(landmarks):
    #mediapipe hands landmarks to numpy array
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


def normalize_landmarks(points):
    #takes np array and normalizes it to range [0,1], so distance from camera doesn't affect the features


    points = points.copy()  # Create a copy to avoid modifying the original array

    #make wrist origin
    wrist = points[WRIST]
    points -= wrist 

    #scale by wrist to base middle knuckle
    middle_mcp = points[MIDDLE_MCP]
    scale = np.linalg.norm(middle_mcp)
    #avoid division by zero
    if scale < 1e-6:
        scale = 1e-6
    points /= scale

    return points

def feature_vector(landmarks):
    #take mediapipe data and convert to 1d vector of normalized features to feed ML model

    raw = landmarks_to_array(landmarks)
    normalized = normalize_landmarks(raw)

    #flatten to 1D array
    return normalized.flatten()


#motion functions

def most_changed_point(sequence):

    #returns the index of the point that has moved the most in a sequence of frames

    all_motion={}
    for i in FINGERTIPS:
        trajectory = sequence[:, i, :2]#only x,y coordinates
        differences = np.diff(trajectory, axis=0)
        all_motion[i] = np.sum(np.linalg.norm(differences, axis=1))
    return max(all_motion, key=all_motion.get)


def _trajectory_features(trajectory):
    """
    trajectory: (T, 2) array of x,y positions for ONE point across the
    whole motion buffer. Returns the same 11 summary numbers that used to
    only get computed for whichever single point moved the most -- this
    is now called once PER FINGERTIP, so every finger's own movement
    pattern gets captured instead of just one "winner" point.
    """
    differences = np.diff(trajectory, axis=0)

    start, end = trajectory[0], trajectory[-1]
    net_motion = end - start
    net_motion_magnitude = np.linalg.norm(net_motion)

    step_lengths = np.linalg.norm(differences, axis=1)
    path_length = float(np.sum(step_lengths))

    straightness = net_motion_magnitude / path_length if path_length > 0 else 0.0

    bounding_box_X = float(np.max(trajectory[:, 0]) - np.min(trajectory[:, 0]))
    bounding_box_Y = float(np.max(trajectory[:, 1]) - np.min(trajectory[:, 1]))

    def reversals(component):
        #count how many times the motion reverses direction in a component (x or y)
        signs = np.sign(component)
        signs = signs[signs != 0]  # Remove zeros to avoid false reversals

        if len(signs) < 2:
            return 0

        return float(np.sum(signs[1:] != signs[:-1]))

    xReversals, yReversals = reversals(differences[:, 0]), reversals(differences[:, 1])

    maxSpeed = float(np.max(step_lengths)) if len(step_lengths) > 0 else 0.0
    meanSpeed = float(np.mean(step_lengths)) if len(step_lengths) > 0 else 0.0

    return [net_motion[0], net_motion[1], net_motion_magnitude,
            path_length, straightness, bounding_box_X, bounding_box_Y,
            xReversals, yReversals, maxSpeed, meanSpeed]


def motion_vector(sequence):
    #returns vector summary of motion for all 5 fingertips 

    sequence = np.array(sequence, dtype=np.float32)

    all_features = []
    for idx in FINGERTIPS:
        trajectory = sequence[:, idx, :2]  # only x,y coordinates
        all_features.extend(_trajectory_features(trajectory))

    return np.array(all_features, dtype=np.float32)


# matches the order of FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
_FINGERTIP_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

_PER_FINGERTIP_FEATURE_NAMES = [
    "net_dx", "net_dy", "net_motion_magnitude",
    "path_length", "straightness",
    "bounding_box_X", "bounding_box_Y",
    "xReversals", "yReversals",
    "maxSpeed", "meanSpeed",
]

#names of vector features in order of the vector returned by motion_vector --
#now 5 fingertips x 11 features each = 55 total, e.g. "thumb_net_dx", "index_path_length", ...
MOTION_FEATURE_NAME_ORDER = [
    f"{finger}_{name}"
    for finger in _FINGERTIP_NAMES
    for name in _PER_FINGERTIP_FEATURE_NAMES
]

#names of static features in order of the vector returned by feature_vector, for example lm0_x
STATIC_FEATURE_NAMES = [
    f"lm{idx}_{axis}" for idx in range(NUM_LANDMARKS) for axis in ("x", "y", "z")
]