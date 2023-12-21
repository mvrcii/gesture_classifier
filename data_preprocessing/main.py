from data_preprocessing.data_preprocessing import DataPreprocessing, DataType
from data_preprocessing.frames_to_sample import FeatureType

if __name__ == "__main__":
    DataPreprocessing(data_type=DataType.TRAINING, feature_type=FeatureType.SYNTHETIC)
    DataPreprocessing(data_type=DataType.VALIDATION, feature_type=FeatureType.SYNTHETIC)
    DataPreprocessing(data_type=DataType.TEST, feature_type=FeatureType.SYNTHETIC)
