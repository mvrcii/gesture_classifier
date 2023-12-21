from data_acquisition.data_acquisition import DataAcquisition, DataType

if __name__ == "__main__":
    # Acquire landmark csv data from video files with full joints
    #DataAcquisition(data_type=DataType.TRAINING)
    #DataAcquisition(data_type=DataType.TRAINING, specific_gesture="pinch")

    #DataAcquisition(data_type=DataType.VALIDATION)

    DataAcquisition(data_type=DataType.TEST)
