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


MODEL_PATH = 'hand_landmarker.task'

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          
    (0, 5), (5, 6), (6, 7), (7, 8),          
    (5, 9), (9, 10), (10, 11), (11, 12),     
    (9, 13), (13, 14), (14, 15), (15, 16),   
    (13, 17), (17, 18), (18, 19), (19, 20),  
    (0, 17),                                 
]

def create_hand_landmarker():
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


def motion_vector(sequence):
    #returns vector summary of motion

    sequence = np.array(sequence, dtype=np.float32)
    activePoint = most_changed_point(sequence)
    trajectory = sequence[:, activePoint, :2]#only x,y coordinates
    differences = np.diff(trajectory, axis=0)

    start,end = trajectory[0], trajectory[-1]
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

        return np.sum(signs[1:] != signs[:-1])

    xReversals, yReversals = reversals(differences[:, 0]), reversals(differences[:, 1])

    maxSpeed = float(np.max(step_lengths)) if len(step_lengths) > 0 else 0.0
    meanSpeed = float(np.mean(step_lengths)) if len(step_lengths) > 0 else 0.0

    return np.array([net_motion[0],net_motion[1],net_motion_magnitude,
                     path_length,straightness,bounding_box_X,bounding_box_Y,
                     xReversals,yReversals,maxSpeed,meanSpeed], dtype=np.float32)


#names  of vector feature in order of the vector returned by motion_vector
MOTION_FEATURE_NAME_ORDER = [
    "net_dx", "net_dy", "net_motion_magnitude",
    "path_length", "straightness",
    "bounding_box_X", "bounding_box_Y",
    "xReversals", "yReversals",
    "maxSpeed", "meanSpeed",
]

#names of static features in order of the vector returned by feature_vector, for example lm0_x
STATIC_FEATURE_NAMES = [
    f"lm{idx}_{axis}" for idx in range(NUM_LANDMARKS) for axis in ("x", "y", "z")
]