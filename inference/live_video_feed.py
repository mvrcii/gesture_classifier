import logging
import pathlib
import queue
import sys
import threading
from collections import defaultdict

import os
import cv2
import mediapipe as mp

from data_acquisition.csv_data_writer import CSVDataWriter
from data_preprocessing.frames_to_sample import SampleFactory, FeatureType
from Prediction_Mode.helper import invalid_landmarks, get_label_and_probability_for_sample
from ml_framework.network import Network
from config import LIVE_INFERENCE_JOINT_MAPPING, INFERENCE_MIN_GESTURE_RATIO, \
    INFERENCE_BLOCK_WINDOW_MULTIPLIER, \
    INFERENCE_MIN_GESTURE_CONFIDENCE, WINDOW_SIZE_MS, DEFAULT_GESTURE_MAPPING

logging.getLogger('tensorflow').setLevel(logging.WARNING)

sys.path.append("../")

# ======================== GLOBAL CONFIG ==========================
label_to_command = {"idle": "idle",
                    "swipe_right": "left",
                    "swipe_left": "right",
                    "swipe_up": "up",
                    "swipe_down": "down",
                    "rotate_right": "rotate_right",
                    "rotate_left": "rotate_left",
                    "spread": "zoom_in",
                    "pinch": "zoom_out",
                    "flip_table": "flip",
                    "spin": "spin",
                    "point": "point"}
# ==================================================================

# find parameters for Pose here: https://google.github.io/mediapipe/solutions/pose.html#solution-apis
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

script_dir = pathlib.Path(__file__).parent


def find_frame_index_ms_ago(timestamps, millis):
    """ Returns the index of the first timestamp (searching backwards from the newest, which is longer than
    WINDOW_SIZE ago, returns -1 if no timestamp with a large enough delta can be found"""
    i = len(timestamps) - 1
    while i >= 0:
        # search backwards from the current frame, up to a frame that was 2 seconds ago
        i = i - 1
        dif = timestamps[-1] - timestamps[i]
        if dif > millis:
            return i
    return -1


TEST_VIDEOS = ["../data_acquisition/training_data/labels/swipe_left/swipe_left_marcel.mp4",  # 0 ========== OURS
               "../data_acquisition/training_data/labels/swipe_right/swipe_right_micha.mp4",  # 1
               "../data_acquisition/training_data/labels/swipe_down/swipe_down_jan.mp4",  # 2
               "../data_acquisition/training_data/labels/swipe_up/swipe_up_micha.mp4",  # 3
               "../data_acquisition/training_data/labels/rotate_left/rotate_left_micha.mp4",  # 4
               "../data_acquisition/training_data/labels/rotate_right/rotate_right_marcel.mp4",  # 5
               "../data_acquisition/training_data/labels/flip_table/flip_table_jan.mp4",  # 6
               "../data_acquisition/training_data/labels/pinch/pinch_marcel.mp4",  # 7
               "../data_acquisition/training_data/labels/spread/spread_micha.mp4",  # 8
               "../data_acquisition/validation_data/labels/swipe_left/swipe_left.mp4",  # 9 ========== VALIDATION
               "../data_acquisition/validation_data/labels/swipe_right/swipe_right.mp4",  # 10
               "../data_acquisition/validation_data/labels/rotate_right/rotate_right.mp4",  # 11
               "../data_acquisition/test_data/labels/swipe_up/swipe_up_emma.mp4",  # 12 ============= EMMA
               "../data_acquisition/test_data/labels/rotate_right/rotate_right_emma.mp4",  # 13
               "../data_acquisition/test_data/labels/swipe_right/swipe_right_emma.mp4",  # 14
               "../data_acquisition/test_data/labels/swipe_left/swipe_left_emma.mp4",  # 15
               ]


class LiveLoopInference:
    def __init__(self, inference_mode=0):
        #self.vid_source = cv2.VideoCapture(filename=str(script_dir.joinpath(TEST_VIDEOS[2])))  # Video
        self.vid_source = cv2.VideoCapture(index=0)  # Webcam

        # ============== Live Inference Modes ==============
        # 0: Slideshow Live Mode
        # 1: Tetris Live Mode

        self.inference_mode = inference_mode

        # TODO: Always use most recent model params
        path = "../training/SYNTHETIC_features/"
        newest_file = path + max(os.listdir(path))
        print(newest_file)
        #self.ml_framework = Network.create_with_hyper_params(newest_file)
        self.network = Network.load(
            "../training/GOOD_CHECKPOINTS/point_spin.npz")
        # self.ml_framework = Network.create_with_hyper_params(
        #     "../Evaluation_Mode/SYNTHETIC_features/hyper_params_2023-03-29_22-04-07.npz")

        self.joint_mapping = LIVE_INFERENCE_JOINT_MAPPING
        self.gesture_mapping = DEFAULT_GESTURE_MAPPING
        # self.gesture_mapping = EVALUATION_GESTURE_MAPPING
        self.feature_type = FeatureType.SYNTHETIC

        self.csv_writer = CSVDataWriter(joint_mapping=self.joint_mapping,
                                        mask=[True, True, True, True])  # Include visibility
        self.sample_factory = SampleFactory(joint_mapping=self.joint_mapping, feature_type=self.feature_type)

        # Video feed variables
        self.show_skeleton = True
        self.show_video = True
        self.print_frame_predictions = True
        self.frame_counter = 0

        # Processing variables
        self.running = True
        self.send_commands = True
        self.detections = []
        self.detection_timestamps = []
        self.last_command_end_pos = -5000
        self.start_blocking = False

        self.command_counts = defaultdict(int)

        self.command_queue = queue.Queue()
        self.thread = threading.Thread(target=self.start)
        self.thread.start()

    def start(self):
        success = True

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.vid_source.isOpened() and success and self.running:
                success, image = self.vid_source.read()

                if not success:
                    break

                self.frame_counter += 1

                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                pose_landmarks = pose.process(image).pose_landmarks

                # Show Video and Skeleton
                if self.show_video:
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    if self.show_skeleton:
                        mp_drawing.draw_landmarks(image=image,
                                                  landmark_list=pose_landmarks,
                                                  connections=mp_pose.POSE_CONNECTIONS,
                                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    cv2.imshow('MediaPipe Pose', image)

                # Handle keyboard events
                key = cv2.waitKey(1) & 0xFF
                if key in [27, ord('q'), ord('Q')]:
                    break

                if cv2.getWindowProperty('MediaPipe Pose', cv2.WND_PROP_VISIBLE) < 1:
                    break

                if invalid_landmarks(pose_landmarks=pose_landmarks):
                    continue

                # Read data
                self.csv_writer.read_data(data=pose_landmarks, timestamp=self.vid_source.get(cv2.CAP_PROP_POS_MSEC))

                # Process data
                self.process_data()
        print(self.command_counts)
        self.vid_source.release()
        cv2.destroyAllWindows()

    def process_data(self):
        # Use all frames between current frame and WINDOW_SIZE ago as input sample, and continue if there are not
        # enough frames to fill up one window
        start_pos = find_frame_index_ms_ago(self.csv_writer.timestamps, WINDOW_SIZE_MS)
        if start_pos == -1:
            return

        # Get the frames for one window
        frames = self.csv_writer.get_frames(start_pos)
        frames.columns = frames.columns.str.lower()

        # Create a sample for the frames
        sample = self.sample_factory.create_sample(frames)

        # Predict label
        detected_label, probability = get_label_and_probability_for_sample(self.network, self.gesture_mapping, sample)

        if self.start_blocking and detected_label == "idle":
            self.start_blocking = False

        debug = ""
        if probability < INFERENCE_MIN_GESTURE_CONFIDENCE and detected_label != "idle":
            debug = " previously " + detected_label
            detected_label = "idle"
        if self.start_blocking:
            detected_label = "idle"
        if self.print_frame_predictions:
            print(f"{detected_label} + {probability}{debug}")
        self.detections.append(detected_label)
        self.detection_timestamps.append(self.csv_writer.timestamps[-1])

        command = self.calculate_command()
        if command is not None:
            print("-------------------------" + command + "------------------------------")
            self.command_counts[command] += 1

            if self.send_commands:
                if self.inference_mode == 0:
                    # Send commands to the command.txt which are received by asynchronous calls from reveal server
                    with open("command.txt", "w") as file:
                        file.write(label_to_command[command])
                elif self.inference_mode == 1:
                    # Put commands in the command queue which is called by the Tetris Loop
                    self.command_queue.put(command)

    def calculate_command(self):
        # Check if the current data fills the window_size and set the start position
        start_pos = find_frame_index_ms_ago(self.detection_timestamps, WINDOW_SIZE_MS)
        if start_pos == -1:
            return

        if (self.csv_writer.timestamps[-1] - self.last_command_end_pos) \
                < INFERENCE_BLOCK_WINDOW_MULTIPLIER * WINDOW_SIZE_MS:
            return

        # Get detections in given time window
        detections = self.detections[start_pos:]

        non_idle_detections = [detection for detection in detections if detection != "idle"]

        if len(non_idle_detections) / len(detections) >= INFERENCE_MIN_GESTURE_RATIO:
            counts = []
            for label in self.gesture_mapping:
                counts.append(non_idle_detections.count(label))

            most_frequent = self.gesture_mapping[counts.index(max(counts))]
            self.last_command_end_pos = self.csv_writer.timestamps[-1]
            print(f"{len(non_idle_detections)}/{len(detections)} => "
                  f"{round(len(non_idle_detections) / len(detections), 2)} \u2265 "
                  f"{INFERENCE_MIN_GESTURE_RATIO}")

            self.start_blocking = True
            self.detections = ['idle' for _ in range(len(self.detections))]
            return most_frequent
        else:
            return None

    def get_command(self):
        # get the latest command from the queue, if any
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None


if __name__ == "__main__":
    LiveLoopInference()
