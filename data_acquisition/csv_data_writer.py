from typing import List

import numpy as np
import pandas as pd
from mediapipe.python.solutions.pose import PoseLandmark
from config import DEFAULT_JOINT_MAPPING, FULL_JOINT_IDs


class CSVDataWriter:
    def __init__(self, joint_mapping: List[PoseLandmark] = DEFAULT_JOINT_MAPPING, mask=(True, True, True, False)):
        self.frame_list = []
        self.timestamps = []
        self.coord_columns = np.array(["x", "y", "z", "confidence"])[np.array(mask)]
        self.joint_names = [joint.name for joint in joint_mapping]  # Transforms PoseLandmark ENUM list to name list
        self.joint_ids = [FULL_JOINT_IDs[joint_name] for joint_name in joint_mapping]
        self.joint_col_names = ["%s_%s" % (joint_name, coord) for joint_name in self.joint_names for coord in
                                self.coord_columns]

    def read_data(self, data, timestamp):
        frame = []
        if data is None:
            return
        for idx in self.joint_ids:
            for attribute in self.coord_columns:
                if attribute == "x":
                    frame.append(data.landmark[idx].x)
                elif attribute == "y":
                    frame.append(data.landmark[idx].y)
                elif attribute == "z":
                    frame.append(data.landmark[idx].z)
                elif attribute == "confidence":
                    frame.append(data.landmark[idx].visibility)
        if timestamp not in self.timestamps:
            self.frame_list.append(frame)
            self.timestamps.append(timestamp)

    def to_csv(self, output_path):
        frames = pd.DataFrame(self.frame_list, columns=self.joint_col_names, index=self.timestamps)
        frames.index.name = "timestamp"
        frames.index = frames.index.astype(int)
        frames.round(5).to_csv(output_path)

    def get_frames(self, start_pos):
        """Returns a sliding window that contains all the relevant data.

        :param start_pos: Start timestamp of the window
        :return: A sliding window as pandas data frame reaching from a given start timestamp up to now (most recent
        frame). The dataframe contains the timestamp as index, as well as all keypoints.
        """
        frames = pd.DataFrame(self.frame_list[start_pos:], columns=self.joint_col_names,
                              index=self.timestamps[start_pos:])
        frames.index.name = "timestamp"
        frames.index = pd.to_timedelta(frames.index, unit='ms')
        return frames
