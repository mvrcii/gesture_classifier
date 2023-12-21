from mediapipe.python.solutions.pose import PoseLandmark

# ================================================ TRAINING SET GENERATION =============================================
WINDOW_SIZE = 1.5
WINDOW_SIZE_MS = WINDOW_SIZE * 1000
FRAMES_PER_SAMPLE = 60  # amount of frames that are extract from one sample (window)
INTERPOLATION_RESAMPLE_RATE = "1ms"

# ------ POSITIVES
POS_SHIFTED_SAMPLES_PER_GESTURE = 2
POS_SHIFT_MAX_PERCENT = 0.1

POS_SPEED_VARIATIONS_PER_GESTURE = 10
POS_SPEED_VARIATION_MAX_PERCENT = 0.3

POS_NOISY_VARIATIONS_PER_SAMPLE = 4
POS_NOISY_VARIATION_MAX_PERCENT = 0.05

POS_SCALED_VARIATIONS_PER_SAMPLE = 1
POS_SCALED_VARIATION_MAX_PERCENT = 0.05

# ------- IDLES
IDL_REGULAR_SAMPLES_PER_FILE = 5

IDL_OVERLAPPING_SAMPLES_PER_FILE = 180
IDL_OVERLAP_MIN_PERCENT = 0.1
IDL_OVERLAP_MAX_PERCENT = 0.3

IDL_STATIC_SAMPLES_PER_FILE = 100

IDL_SCALED_VARIATIONS_PER_SAMPLE = 0
IDL_SCALED_VARIATION_MAX_PERCENT = 0.05

IDL_NOISY_VARIATIONS_PER_SAMPLE = 1
IDL_NOISY_VARIATION_MAX_PERCENT = 0.05

# ======================================================================================================================

# Gestures for overall purpose
# IMPORTANT: 'idle' is contained!
DEFAULT_GESTURE_MAPPING = [
    "idle",
    "swipe_right",
    "swipe_left",
    "rotate_right",
    "rotate_left",
    "swipe_up",
    "swipe_down",
    "spread",
    "pinch",
    "flip_table",
    "spin",
    "point"
]

EVALUATION_GESTURE_MAPPING = [
    "idle",
    "swipe_right",
    "swipe_left",
    "rotate"
]

# Gestures to be included in the dataset creation process
# IMPORTANT: Only use contiguous gestures in ascending order (see gestures.yml) starting at swipe right
DATASET_CREATION_GESTURES = [
    "swipe_right",
    "swipe_left",
    "rotate_right",
    "rotate_left",
    "swipe_up",
    "swipe_down",
    "spread",
    "pinch",
    "flip_table",
    "spin",
    "point"
]

SYNTHETIC_FEATURE_NAMES = ["delta_x_left", "delta_y_left", "delta_z_left", "delta_x_right", "delta_y_right",
                           "delta_z_right", "delta_el_x_left", "delta_el_y_left", "delta_el_x_right",
                           "delta_el_y_right", "delta_wel_x_left", "delta_wel_y_left", "delta_wel_x_right",
                           "delta_wel_y_right", "wrist_dist"]

FULL_JOINT_IDs = {PoseLandmark.NOSE: 0,
                  PoseLandmark.LEFT_EYE_INNER: 1,
                  PoseLandmark.LEFT_EYE: 2,
                  PoseLandmark.LEFT_EYE_OUTER: 3,
                  PoseLandmark.RIGHT_EYE_INNER: 4,
                  PoseLandmark.RIGHT_EYE: 5,
                  PoseLandmark.RIGHT_EYE_OUTER: 6,
                  PoseLandmark.LEFT_EAR: 7,
                  PoseLandmark.RIGHT_EAR: 8,
                  PoseLandmark.MOUTH_LEFT: 9,
                  PoseLandmark.MOUTH_RIGHT: 10,
                  PoseLandmark.LEFT_SHOULDER: 11,
                  PoseLandmark.RIGHT_SHOULDER: 12,
                  PoseLandmark.LEFT_ELBOW: 13,
                  PoseLandmark.RIGHT_ELBOW: 14,
                  PoseLandmark.LEFT_WRIST: 15,
                  PoseLandmark.RIGHT_WRIST: 16,
                  PoseLandmark.LEFT_PINKY: 17,
                  PoseLandmark.RIGHT_PINKY: 18,
                  PoseLandmark.LEFT_INDEX: 19,
                  PoseLandmark.RIGHT_INDEX: 20,
                  PoseLandmark.LEFT_THUMB: 21,
                  PoseLandmark.RIGHT_THUMB: 22,
                  PoseLandmark.LEFT_HIP: 23,
                  PoseLandmark.RIGHT_HIP: 24,
                  PoseLandmark.LEFT_KNEE: 25,
                  PoseLandmark.RIGHT_KNEE: 26,
                  PoseLandmark.LEFT_ANKLE: 27,
                  PoseLandmark.RIGHT_ANKLE: 28,
                  PoseLandmark.LEFT_HEEL: 29,
                  PoseLandmark.RIGHT_HEEL: 30,
                  PoseLandmark.LEFT_FOOT_INDEX: 31,
                  PoseLandmark.RIGHT_FOOT_INDEX: 32}

FULL_JOINT_MAPPING = [joint for joint in list(FULL_JOINT_IDs)]

DEFAULT_JOINT_MAPPING = [PoseLandmark.LEFT_WRIST,
                         PoseLandmark.RIGHT_WRIST,
                         PoseLandmark.LEFT_ELBOW,
                         PoseLandmark.RIGHT_ELBOW,
                         PoseLandmark.LEFT_SHOULDER,
                         PoseLandmark.RIGHT_SHOULDER]

LIVE_INFERENCE_JOINT_MAPPING = [PoseLandmark.LEFT_WRIST,
                                PoseLandmark.RIGHT_WRIST,
                                PoseLandmark.LEFT_ELBOW,
                                PoseLandmark.RIGHT_ELBOW,
                                PoseLandmark.LEFT_SHOULDER,
                                PoseLandmark.RIGHT_SHOULDER,
                                PoseLandmark.LEFT_EYE,
                                PoseLandmark.RIGHT_EYE]

# ============== NETWORK CONFIG ==============
NETWORK_SEED = 12  # -1 for using random seed
NETWORK_HIDDEN_LAYERS_SHAPE = [18]

# ============== TRAINING CONFIG ==============
ADJUST_ALPHA = False
TRAINING_ITERATIONS = 500
TRAINING_ALPHA = 0.00003
TRAINING_LOG_INTERVALS = {'first': 1, 'second': 10}
TRAINING_USE_FEATURE_SCALING = True
TRAINING_USE_REGULARIZATION = True
TRAINING_LAMBDA = 0.0001
PCA_THRESHOLD = 0.99

# ============== PLOTTING CONFIG ==============
PLOT_CONF_MAT_2D = True

# ============== LIVE VIDEO INFERENCE CONFIG ==============
INFERENCE_BLOCK_WINDOW_MULTIPLIER = 1.5
INFERENCE_MIN_GESTURE_CONFIDENCE = 0.8

INFERENCE_MIN_GESTURE_RATIO = 0.01
INFERENCE_MIN_WRIST_VISIBILITY_THRESHOLD = 0.5
INFERENCE_MIN_SHOULDER_DISTANCE_TO_ARM_LENGTH_RATIO = 0.5
