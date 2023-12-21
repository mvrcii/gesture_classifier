import os
import time
from os.path import exists

import numpy as np
from matplotlib import pyplot as plt

from config import *
from data_preprocessing.data_preprocessing import DataType
from ml_framework.network import Network


def load_data(data_type: DataType, feature_type):
    """Loads data from a given 'Data Preprocessing' source directory."""
    src_dir = data_type.name.lower() + '_data'

    X_path = f"Data_Preprocessing/{src_dir}/{feature_type.name}_features/X.npy"
    Y_path = f"Data_Preprocessing/{src_dir}/{feature_type.name}_features/Y.npy"
    if not exists(X_path) or not exists(Y_path):
        raise FileNotFoundError(
            f"For the given directory Data_Preprocessing/{src_dir}/{feature_type.name}_features/ either X.npy "
            f"or Y.npy does not exist.")
    X = np.load(X_path, allow_pickle=True)
    Y = np.load(Y_path, allow_pickle=True)
    return X, Y


def test_network(network, feature_type):
    X_test, y_test = load_data(data_type=DataType.TEST, feature_type=feature_type)

    # Predict
    y_test_pred = network.predict(X=X_test, apply_feature_scaling=True)

    # Evaluate
    result = network.evaluate(y_pred=y_test_pred, ground_truth=y_test)
    plt.show()

    return result


def train(feature_type, train_data_dir, val_split=0.2, model="", show_plots=True):
    # Directory to save model params
    target_dir = f"Training/{feature_type.name}_features"
    class_labels = DEFAULT_GESTURE_MAPPING
    class_cnt = len(class_labels)

    # Load the training data
    X, Y = load_data(data_type=DataType.TRAINING, feature_type=feature_type)

    # Shuffle data
    np.random.seed(NETWORK_SEED)
    perm = np.random.permutation(len(X))
    X = X[perm]
    Y = Y[perm]

    # Perform Validation Split
    split_index = int(len(X) * val_split)
    X_train = X[split_index:]
    Y_train = Y[split_index:]
    X_val = X[:split_index]
    Y_val = Y[:split_index]

    # Transfer Learning: Load existing model
    if model:
        model_path = f"{target_dir}/{model}"
        network = Network.load(source_dir=model_path)
    else:
        layer_shape = [X.shape[1]]
        for layer in NETWORK_HIDDEN_LAYERS_SHAPE:
            layer_shape.append(layer)
        layer_shape.append(class_cnt)

        # Create a new model
        network = Network(layer_shape=layer_shape, seed=NETWORK_SEED)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    # Train the model
    network.train(
        hidden_layer_shape=NETWORK_HIDDEN_LAYERS_SHAPE,
        iterations=TRAINING_ITERATIONS,
        alpha=TRAINING_ALPHA,
        X_train=X_train,
        y_train=Y_train,
        X_val=X_val,
        y_val=Y_val,
        show_plots=show_plots,
        log_intervals=TRAINING_LOG_INTERVALS,
        use_feature_scaling=TRAINING_USE_FEATURE_SCALING,
        lambda_regularization=TRAINING_LAMBDA,
        use_regularization=TRAINING_USE_REGULARIZATION,
        class_labels=class_labels,
        adjust_alpha=ADJUST_ALPHA,
        pca_threshold=PCA_THRESHOLD,
    )

    path = os.path.join('training', f'{feature_type.name}_features', f'confmat2d_{timestamp}.mp4')
    network.create_confusion_matrix_2d_live_mp4(path)

    network.save(
        target_dir=target_dir,
        timestamp=timestamp
    )

    return network
