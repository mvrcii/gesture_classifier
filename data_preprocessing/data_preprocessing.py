import multiprocessing
import os
import random
import time
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as config
from data_preprocessing.frames_to_sample import SampleFactory, FeatureType
from config import *

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)


def report_on_dataset(all_labels, all_samples, result_dir, feature_type):
    unique, counts = np.unique(all_labels, return_counts=True)
    report = "=======================DATASET REPORT==============================\n"
    report += f"FEATURE TYPE: {feature_type.name}\n"
    report += f"TOTAL: {len(all_samples)} samples.\n"

    mapping = DEFAULT_GESTURE_MAPPING

    for elem, count in zip(unique, counts):
        report += f"{mapping[elem]}: {count} samples ({round(count / len(all_labels) * 100, 2)}%)\n"
    print(report)
    with open(f"{result_dir}/report.txt", "w") as file:
        file.write(report)


def get_label(gesture_mapping, file_name):
    label = -1
    for idx, gesture in enumerate(gesture_mapping):
        if gesture in file_name:
            label = idx
    if label == -1:
        print('\033[31m' + ' Error trying to get label for filename' + '\033[0m')
        exit(1)
    return label


def get_dataset_files_from_path(source_path):
    file_names = os.listdir(source_path)
    files_to_use = []
    for file in file_names:
        for gesture in DATASET_CREATION_GESTURES:
            if gesture in file:
                files_to_use.append(file)
    return files_to_use


def handle_error(e):
    print(f"Error in worker process: {e}")


class DataType(Enum):
    TRAINING = 1
    VALIDATION = 2
    TEST = 3


class DataPreprocessing:
    def __init__(self,
                 data_type: DataType,
                 feature_type:
                 FeatureType,
                 joint_mapping=DEFAULT_JOINT_MAPPING,
                 gesture_mapping=DEFAULT_GESTURE_MAPPING):
        """The data preprocessing module which takes the data as csv file and creates X (samples) and Y (labels) for it.

        :param data_type: The directory where the labelled csv data is located.
        :param feature_type: The feature handling which should be used.
        :param joint_mapping: The joint mapping which should be used.
        :param gesture_mapping: The gesture mapping which should be used. Important: should contain 'idle'!
        """

        if not isinstance(data_type, DataType):
            raise ValueError("Invalid data type value provided")
        self.data_type = data_type.name.lower() + '_data'

        self.data_source_dir = f"../Data_Acquisition/{self.data_type}"
        self.result_dir = f"/{self.data_type}/{feature_type.name}_features"
        self.labelled_path = self.data_source_dir + "/csv_data_labelled"

        self.feature_type = feature_type

        if data_type == DataType.TRAINING:
            self.data_augmentation = True
        else:
            self.data_augmentation = False

        self.gesture_mapping = gesture_mapping

        self.sample_factory = SampleFactory(joint_mapping=joint_mapping, feature_type=feature_type)

        self.start()

    def start(self):
        time_start = time.time()

        # Generate Samples and Labels for all files
        all_samples = []
        all_labels = []

        cpu_count = multiprocessing.cpu_count()
        results = multiprocessing.Manager().dict()

        files_to_use = get_dataset_files_from_path(f"{self.labelled_path}/")
        files_left = files_to_use.copy()

        # self.data_for_file(training=training, file_name="rotate_right_micha.csv")
        # exit()

        while len(results) < len(files_to_use):
            print(f"Amount of files to use: {len(files_to_use)}")
            num_threads = min(len(files_left), cpu_count)
            pool = multiprocessing.Pool(num_threads)
            print(f"Creating pool of size {num_threads}")
            for _ in range(num_threads):
                file_name = files_left.pop(0)
                print("Creating worker for " + file_name)
                pool.apply_async(self.data_for_file, args=(results, file_name), error_callback=handle_error)
            pool.close()
            pool.join()
            len_res = len(results)
            if len_res == 0:
                raise AssertionError(
                    "Results empty. Debug data preprocessing for a single file to get detailed information.")
            else:
                print(len(results))
        print("Processing done")

        for _, value in results.items():
            all_samples.extend(value["positive_samples"])
            all_samples.extend(value["idle_samples"])
            all_labels.extend(value["positive_labels"])
            all_labels.extend(value["idle_labels"])

        self.save(np.array(all_samples), np.array(all_labels))
        time_end = time.time()
        print("Time taken for data preprocessing: ", round(time_end - time_start, 2), "s")

    @staticmethod
    def map_labels_for_eval_mode(labels, mapping: []):
        new_labels = []
        for label in labels:
            label_str = DEFAULT_GESTURE_MAPPING[label]
            # If label still exists, add it with (new) correct index
            if label_str in mapping:
                new_labels.append(mapping.index(label_str))
            # If label is 'rotate_right', add it with 'rotate' index
            elif label_str == 'rotate_right':
                new_labels.append(mapping.index('rotate'))
            # If label does not exist, add it with 'idle' index
            else:
                new_labels.append(mapping.index('idle'))
        if np.unique(new_labels).size != len(mapping):
            raise AssertionError(
                "The dimension of the mapped labels does not match the dimension of the given mapping.")
        return new_labels

    def data_for_file(self, results, file_name):
        samples_for_file = {}

        # determine correct label for positive samples of this file
        label: int = get_label(self.gesture_mapping, file_name)

        # read and adjust file data, set global frames variable to Pandas DF containing frames
        frames: pd.DataFrame = self.read_frames_for_file(file_name)
        frames.columns = frames.columns.str.lower()

        # ====================================== Extract Samples =============================================
        # POSITIVE SAMPLES
        positive_samples = self.generate_all_positive_samples(frames, file_name, label)
        num_features = int(len(positive_samples[0].flatten()) / FRAMES_PER_SAMPLE)

        # IDLE SAMPLES
        idle_samples = self.generate_all_idle_samples(frames, num_features, positive_samples)

        # ========= Create Labels =========
        positive_labels = [label for _ in range(len(positive_samples))]
        idle_labels = [DEFAULT_GESTURE_MAPPING.index("idle") for _ in range(len(idle_samples))]

        # Append samples and labels to the lists of samples and labels for all files
        samples_for_file["positive_samples"] = positive_samples
        samples_for_file["idle_samples"] = idle_samples
        samples_for_file["positive_labels"] = positive_labels
        samples_for_file["idle_labels"] = idle_labels
        results[file_name] = samples_for_file

    @staticmethod
    def get_gesture(string):
        # Split the string into words
        words = string.split("_")

        # Check if the first word matches any of the gestures in DATASET_CREATION_GESTURES
        for gesture in DATASET_CREATION_GESTURES:
            if gesture.startswith(words[0]):
                return gesture
        return None

    def generate_all_positive_samples(self, frames, filepath, label):
        """Generates all positive samples.

        :return: A list of all positive samples.
        """
        filename, _ = os.path.splitext(filepath)
        gesture = self.gesture_mapping[label]

        labels = pd.read_csv(f"{self.data_source_dir}/labels/{gesture}/{filename}.txt",
                             sep="\t",
                             header=None,
                             usecols=[3, 5, 8],
                             names=["start", "end", "label"])
        labels["start"] = pd.to_timedelta(labels["start"], unit="s")
        labels["end"] = pd.to_timedelta(labels["end"], unit="s")

        # POSITIVE SAMPLES
        positive_samples = self.fast_extract_positive_samples(frames, labels)
        # print(f"Added {len(positive_samples)} regular positive samples")
        if not self.data_augmentation:
            return positive_samples

        # =================== DATA AUGMENTATION ===================

        # SPEED VARIATIONS
        for _ in range(POS_SPEED_VARIATIONS_PER_GESTURE):
            speed_varied_samples = self.fast_extract_positive_samples(frames, labels,
                                                                      vary_speed_range=POS_SPEED_VARIATION_MAX_PERCENT)
            positive_samples.extend(speed_varied_samples)
            # print(f"Added {len(speed_varied_samples)} speed varied positive samples")

        # SHIFTED SAMPLES
        for _ in range(POS_SHIFTED_SAMPLES_PER_GESTURE):
            shifted_samples = self.fast_extract_positive_samples(frames, labels, offset_range=POS_SHIFT_MAX_PERCENT)
            positive_samples.extend(shifted_samples)
            # print(f"Added {len(speed_varied_samples)} speed varied positive samples")

        # RANDOM VARIATIONS SCALED
        variations = []
        for p in positive_samples:
            variations.extend(
                self.create_variations_scaled(p, POS_SCALED_VARIATIONS_PER_SAMPLE, POS_SCALED_VARIATION_MAX_PERCENT))
        # print(f"Added {len(variations)} scaled varied positive samples")
        positive_samples.extend(variations)

        # RANDOM VARIATIONS
        variations = []
        for p in positive_samples:
            variations.extend(
                self.create_variations_noisy(p, POS_NOISY_VARIATIONS_PER_SAMPLE, POS_NOISY_VARIATION_MAX_PERCENT))
        # print(f"Added {len(variations)} noisy varied positive samples")
        positive_samples.extend(variations)

        return positive_samples

    def fast_extract_positive_samples(self, frames, labels, vary_speed_range: float = 0.0, offset_range: float = 0.0):
        """
        Extracts 1 positive sample for each annotated gesture in the passed frames
        :param frames:
        :param labels:
        :param vary_speed_range: percent by which window size will be change randomly (e.g. 0.05 => window size
        can be changed randomly between 95% and 105%)
        :param offset_range: value between 0 = no shift and 1 = window shifted randomly forward or
         backwards up to 100% of the window_size (e.g. 0.8 => window can be shifted up to 80% of window size forward
         or backward)
        :return:
        """
        positives = []
        for _, label in labels.iterrows():
            window_size = config.WINDOW_SIZE
            start = label["start"]
            stop = label["end"]
            delta = (stop - start)
            middle_ts = start + (delta * 0.5)

            # for data augmentation, randomly shift window forwards or backwards
            if offset_range != 0.0:
                middle_ts += pd.to_timedelta(window_size * np.random.uniform(-offset_range, offset_range), unit="s")

            # For data augmentation, randomly decrease or increase window size by some percentage, to stretch / squish
            # sample
            if vary_speed_range != 0.0:
                window_size *= np.random.uniform((1 - vary_speed_range), (1 + vary_speed_range))

            bottom_border = middle_ts - pd.to_timedelta(window_size * 0.5, unit="s")
            top_border = middle_ts + pd.to_timedelta(window_size * 0.5, unit="s")
            sample_frames = frames.loc[(frames.index >= bottom_border) & (frames.index <= top_border)]

            sample = self.sample_factory.create_sample(sample_frames)
            positives.append(sample)
        return positives

    @staticmethod
    def generate_static_idle_samples_from_positives(positive_samples, num_features):
        if IDL_STATIC_SAMPLES_PER_FILE == 0:
            return []
        pos_copy = positive_samples.copy()
        static_idle_samples = []
        for _ in range(IDL_STATIC_SAMPLES_PER_FILE):
            s = random.choice(pos_copy)
            s = s.reshape(FRAMES_PER_SAMPLE, num_features)
            frame_idx = random.randint(0, len(s) - 1)
            frame = s[frame_idx]
            sample = np.tile(frame, (config.FRAMES_PER_SAMPLE, 1))
            static_idle_samples.append(sample.flatten())
        return static_idle_samples

    def generate_all_idle_samples(self, frames, num_features, positive_samples):
        idle_samples = self.extract_idle_samples(frames)

        # print(f"Added {len(idle_samples)} regular idle samples")
        if not self.data_augmentation:
            return idle_samples

        # =================== DATA AUGMENTATION
        # OVERLAPPING IDLE SAMPLES
        overlapping_idle_samples = self.extract_overlapping_idle_samples(frames, IDL_OVERLAPPING_SAMPLES_PER_FILE,
                                                                         IDL_OVERLAP_MIN_PERCENT,
                                                                         IDL_OVERLAP_MAX_PERCENT)
        # print(f"Added {len(overlapping_idle_samples)} overlapping idle samples")
        idle_samples.extend(overlapping_idle_samples)

        # STATIC POSITIVE SAMPLES
        static_positive_samples = self.generate_static_idle_samples_from_positives(positive_samples, num_features)
        # print(f"Added {len(static_positive_samples)} static idle samples")
        idle_samples.extend(static_positive_samples)

        # RANDOM SCALED IDLE SAMPLE VARIATIONS
        variations = []
        for idle_sample in idle_samples:
            variations.extend(
                self.create_variations_noisy(idle_sample, IDL_SCALED_VARIATIONS_PER_SAMPLE,
                                             IDL_SCALED_VARIATION_MAX_PERCENT))
        # print(f"Added {len(variations)} random scaled varied idle samples")
        idle_samples.extend(variations)

        # RANDOM NOISY IDLE SAMPLE VARIATIONS
        variations = []
        for idle_sample in idle_samples:
            variations.extend(
                self.create_variations_noisy(idle_sample, IDL_NOISY_VARIATIONS_PER_SAMPLE,
                                             IDL_NOISY_VARIATION_MAX_PERCENT))
        # print(f"Added {len(variations)} random noisy varied idle samples")
        idle_samples.extend(variations)

        return idle_samples

    def extract_idle_samples(self, frames):
        idles = []
        j = -1
        while j < len(frames):
            j = j + 1
            if frames.iloc[j]["ground_truth"] == "idle":
                start = pd.to_timedelta(frames.iloc[j].name, unit="s")
                while frames.iloc[j]["ground_truth"] == "idle":
                    j = j + 1
                    if j >= len(frames) - 1:
                        return random.sample(idles, min(IDL_REGULAR_SAMPLES_PER_FILE, len(idles)))
                    stop = pd.to_timedelta(frames.iloc[j].name, unit="s")
                    diff = stop - start
                    if diff.total_seconds() > WINDOW_SIZE:
                        sample_frames = frames.loc[(frames.index >= start) & (frames.index <= stop)]
                        sample = self.sample_factory.create_sample(sample_frames)
                        idles.append(sample)
                        break
        return random.sample(idles, min(IDL_REGULAR_SAMPLES_PER_FILE, len(idles)))

    def extract_overlapping_idle_samples(self, frames, how_many, min_overlap_percentage, max_overlap_percentage):
        idles = []
        estimated_fps = 30
        total_frames = len(frames)
        while len(idles) < how_many:
            random_start_pos = random.randint(0, total_frames - int(estimated_fps * WINDOW_SIZE))
            start_time = pd.to_timedelta(frames.iloc[random_start_pos].name, unit="s")
            end_pos = random_start_pos
            while end_pos < len(frames) - 1:
                end_pos += 1
                end_time = pd.to_timedelta(frames.iloc[end_pos].name, unit="s")
                diff = end_time - start_time
                if diff.total_seconds() > WINDOW_SIZE:
                    sample_frames = frames.loc[(frames.index >= start_time) & (frames.index <= end_time)]
                    idle_frame_count = sample_frames[sample_frames["ground_truth"] == "idle"].shape[0]
                    idle_percentage = idle_frame_count / len(sample_frames)
                    overlap_percentage = 1 - idle_percentage
                    if min_overlap_percentage <= overlap_percentage <= max_overlap_percentage:
                        sample = self.sample_factory.create_sample(sample_frames)
                        idles.append(sample)
                    break
        return idles

    @staticmethod
    def create_variations_noisy(flattened_sample, amount, percent_random_change):
        variants = []
        for _ in range(amount):
            change = np.random.uniform((1 - percent_random_change), (1 + percent_random_change),
                                       size=flattened_sample.shape)
            variants.append(flattened_sample * change)
        return variants

    @staticmethod
    def create_variations_scaled(flattened_sample, amount, percent_random_change):
        variants = []
        for _ in range(amount):
            change = np.random.uniform((1 - percent_random_change), (1 + percent_random_change))
            variants.append(flattened_sample * change)
        return variants

    def read_frames_for_file(self, file_name) -> pd.DataFrame:
        dataFrame = pd.read_csv(f"{self.labelled_path}/{file_name}")
        dataFrame["timestamp"] = pd.to_timedelta(dataFrame["timestamp"], unit="ms")
        dataFrame = dataFrame.set_index("timestamp")
        dataFrame.index = dataFrame.index.rename("timestamp")
        return dataFrame

    def save(self, samples: np.array, labels: np.array):
        # Create directory if necessary
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        print("Saving samples and labels\n")
        np.save(f"{self.result_dir}/X.npy", samples)
        np.save(f"{self.result_dir}/Y.npy", labels)
        report_on_dataset(labels, samples, self.result_dir, self.feature_type)

    @staticmethod
    def debug_visualize_sample(sample, down_sampled, feature):
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.bar(range(len(sample)), sample[feature])
        ax2.bar(range(len(down_sampled)), down_sampled[feature])
        plt.show()
