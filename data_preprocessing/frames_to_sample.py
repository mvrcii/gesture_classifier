import enum
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mediapipe.python.solutions.pose import PoseLandmark

from config import INTERPOLATION_RESAMPLE_RATE, FRAMES_PER_SAMPLE


class FeatureType(enum.Enum):
    RAW = 1,
    SYNTHETIC = 2


def debug_visualize_sample(sample):
    plt.bar(range(len(sample)), sample)
    plt.show()


def split_positive_negative(dataframe):
    return [dataframe.clip(upper=0).abs(), dataframe.clip(lower=0)]


def splits_for_list(df_list):
    result = []
    for df in df_list:
        result.extend(split_positive_negative(df))
    return result


class SampleFactory(object):

    def __init__(self, joint_mapping: List[PoseLandmark], feature_type: FeatureType = FeatureType.RAW):
        coord_columns = np.array(["x", "y", "z", "confidence"])[np.array([True, True, True, False])]
        joint_names = [joint.name for joint in joint_mapping]
        self.joint_col_names = ["%s_%s" % (joint_name, coord) for joint_name in joint_names for coord in
                                coord_columns]
        self.feature_type = feature_type

    def create_sample(self, frames: pd.DataFrame):
        interpolated_frames = interpolate(frames)
        return self.synthetic_feature_processing(interpolated_frames)

    def raw_feature_processing(self, interpolated_frames):
        frames = np.array(interpolated_frames[self.joint_col_names])
        return frames.flatten()

    @staticmethod
    def synthetic_feature_processing(interpolated_frames):
        # deltas x / y, shoulder to wrists
        delta_x_left = interpolated_frames["left_shoulder_x"] - interpolated_frames["left_wrist_x"]
        delta_y_left = interpolated_frames["left_shoulder_y"] - interpolated_frames["left_wrist_y"]
        delta_x_right = interpolated_frames["right_shoulder_x"] - interpolated_frames["right_wrist_x"]
        delta_y_right = interpolated_frames["right_shoulder_y"] - interpolated_frames["right_wrist_y"]

        # deltas x / y, shoulder to elbow
        delta_el_x_left = interpolated_frames["left_shoulder_x"] - interpolated_frames["left_elbow_x"]
        delta_el_y_left = interpolated_frames["left_shoulder_y"] - interpolated_frames["left_elbow_y"]
        delta_el_x_right = interpolated_frames["right_shoulder_x"] - interpolated_frames["right_elbow_x"]
        delta_el_y_right = interpolated_frames["right_shoulder_y"] - interpolated_frames["right_elbow_y"]

        # deltas x / y, wrist to elbow
        delta_wel_x_left = interpolated_frames["left_wrist_x"] - interpolated_frames["left_elbow_x"]
        delta_wel_y_left = interpolated_frames["left_wrist_y"] - interpolated_frames["left_elbow_y"]
        delta_wel_x_right = interpolated_frames["right_wrist_x"] - interpolated_frames["right_elbow_x"]
        delta_wel_y_right = interpolated_frames["right_wrist_y"] - interpolated_frames["right_elbow_y"]

        # deltas z shoulder to wrists
        delta_z_left = interpolated_frames["left_shoulder_z"] - interpolated_frames["left_wrist_z"]
        delta_z_right = interpolated_frames["right_shoulder_z"] - interpolated_frames["right_wrist_z"]

        delta_x_wrist = interpolated_frames["left_wrist_x"] - interpolated_frames["right_wrist_x"]
        delta_y_wrist = interpolated_frames["left_wrist_y"] - interpolated_frames["right_wrist_y"]
        delta_x_wrist = delta_x_wrist.apply(lambda x: x ** 2)
        delta_y_wrist = delta_y_wrist.apply(lambda x: x ** 2)
        wrist_dist = np.sqrt(delta_x_wrist + delta_y_wrist)

        # TODO: Maybe use helpful feature combinations x * y (ODER SO)
        # TODO: Try: Augmentation with Scaling
        # TODO: Try: In Live-Mode adjust Features to Distance

        # use column_stack instead of array here, to keep the shape of rows representing frames and columns representing
        # features, e.g. input features = (60, 133), result when using column_stack = (60, 10) (instead of (10, 60))
        # this would not impact the final result since the samples are flattened anyway, however it makes more sense
        # semantically and helps debugging
        result = np.column_stack(
            [delta_x_left, delta_y_left, delta_z_left, delta_x_right, delta_y_right, delta_z_right, delta_el_x_left,
             delta_el_y_left, delta_el_x_right, delta_el_y_right, delta_wel_x_left, delta_wel_y_left, delta_wel_x_right,
             delta_wel_y_right, wrist_dist])
        return result.flatten()


def interpolate(frames: pd.DataFrame):
    """Takes a pandas data frame, interpolates its values according to a given resampling rate, and
    finally returns the desired amount of frames for one sample."""

    # Separate numerical and non-numerical columns
    numeric_frames = frames.select_dtypes(include=[float])
    non_numeric_frames = frames.select_dtypes(exclude=[float])

    # Interpolate only the numerical columns
    interpolated_numeric = numeric_frames.resample(INTERPOLATION_RESAMPLE_RATE).interpolate(method='linear')
    # Due to interpolation, the non-numerical data might have missing entries. Forward fill to handle that
    interpolated_non_numeric = non_numeric_frames.resample(INTERPOLATION_RESAMPLE_RATE).ffill().bfill()

    interpolated = pd.concat([interpolated_numeric, interpolated_non_numeric], axis=1)

    indices = np.linspace(0, len(interpolated) - 1, FRAMES_PER_SAMPLE)
    down_sampled = interpolated.iloc[indices.astype(int)]  # down sampling
    return down_sampled
