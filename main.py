from data_preprocessing.frames_to_sample import FeatureType
from ml_framework.network import Network
from training.train import train, test_network

if __name__ == "__main__":
    #network = train(train_data_dir="training_data", feature_type=FeatureType.SYNTHETIC, show_plots=True)

    network = Network.load("training/SYNTHETIC_features/hyper_params_2023-10-21_18-00-42.npz")
    test_network(network=network, feature_type=FeatureType.SYNTHETIC)
