import inspect
import os
import random
import subprocess
import sys
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

import config

np.set_printoptions(threshold=sys.maxsize)


def fit(X: np.array) -> Tuple[np.array, np.array]:
    """Calculates the mean and standard deviation of each feature of the input data.

    :param X: A numpy array containing the input data.
    :return: A tuple containing the mean and standard deviation of each feature of the input data.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 0.00001  # Avoid Division-By-Zero errors
    return mean, std


def transform(X: np.array, t_mat: np.array) -> np.array:
    """Transforms the input data into a lower-dimensional space using the given PCA transformation matrix.

     :param X: A numpy array containing the input data to be transformed.
     :param t_mat: A numpy array containing the PCA transformation matrix.
     :return: A numpy array containing the transformed data.
     """
    return np.dot(X, t_mat)


def standardize(X: np.array, mean: np.array, std: np.array) -> np.array:
    """Standardizes the input data using the mean and standard deviation calculated from the training data.

    :param X: A numpy array containing the input data to be standardized.
    :param mean: A numpy array containing the mean of each feature of the training data.
    :param std: A numpy array containing the standard deviation of each feature of the training data.
    :return: A numpy array containing the standardized input data.
    """
    return (X - mean) / std


def add_bias(X: np.array) -> np.array:
    """Adds a bias term to the input data.

    :param X: A numpy array containing the input data.
    :return: A numpy array containing the input data with an additional bias column.
    """
    bias = 1 * np.ones((X.shape[0], 1))
    return np.hstack((bias, X))


def calc_pca_matrix(X_std: np.array,
                    threshold: float = 0.95,
                    show_plots: bool = False) -> np.array:
    """Calculates a transformation matrix for a given dataset using principal component analysis with a given threshold.

    :param X_std: A numpy array containing the standardized dataset.
    :param threshold: A float representing the threshold for the variance. Default value is 0.95.
    :param show_plots: A boolean representing if the training should be plotted. Default value is False.
    :return: A numpy array representing the transformation matrix to transform datasets based on the performed PCA.
    """

    # Step 1: Calculate the covariance matrix
    cov_mat = np.cov(X_std, rowvar=False)

    # Step 2: Calculate the eigenvalues and eigenvectors
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

    # Step 3: Sort the eigenvalues and eigenvectors
    idx = eigen_vals.argsort()[::-1]
    eigen_vals = eigen_vals[idx]
    eigen_vecs = eigen_vecs[:, idx]

    # Step 4: Choose the number of principal components
    cumulative_var_exp = np.cumsum(eigen_vals) / np.sum(eigen_vals)
    n_components = np.argmax(cumulative_var_exp >= threshold) + 1

    # Step 5: Calculate the transformation matrix for the given threshold (n_components)
    transformation_matrix = eigen_vecs[:, :n_components]

    # Plot the cumulative explained variance ratio
    if show_plots:
        idx_threshold = np.argmax(cumulative_var_exp >= threshold)
        val_threshold = cumulative_var_exp[idx_threshold]

        fig2, ax = plt.subplots(figsize=(8, 6))
        ax.plot(cumulative_var_exp, linewidth=2, color='blue')
        ax.set_xlabel('Number of components', fontsize=12)
        ax.set_ylabel('Cumulative explained variance', fontsize=12)
        ax.set_title('Cumulative explained variance ratio', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xlim([0, idx_threshold * 1.25])  # limit x-axis to 200 components
        ax.axhline(y=threshold, linestyle='--', color='black', alpha=0.5)
        ax.annotate(f'{threshold:.0%} threshold', xy=(80, threshold - 0.04), fontsize=12, color='black')
        ax.axvline(x=idx_threshold, linestyle='--', color='black', alpha=0.5)
        ax.annotate(f'{idx_threshold} components for {threshold:.0%} variance', xy=(idx_threshold + 5, 0.5),
                    fontsize=12,
                    color='black')
        ax.scatter(idx_threshold, val_threshold, marker='o', s=100, facecolors='none', edgecolors='red')
        ax.tick_params(axis='x', which='minor', length=5)

    return transformation_matrix


def calc_per_class_scores(ground_truth, pred_labels, class_count):
    recall_p_class, precision_p_class, f1_p_class = [], [], []
    for class_idx in range(class_count):
        cm = calc_confusion_matrix(ground_truth=ground_truth, predicted_labels=pred_labels, class_idx=class_idx)
        recall_p_class.append(calc_recall(cm))
        precision_p_class.append(calc_precision(cm))
        f1_p_class.append(calc_f1(cm))
    return f1_p_class, precision_p_class, recall_p_class


def calculate_accuracy(prediction, ground_truth):
    """Calculates the total accuracy.

    :param prediction: The raw prediction as numpy array.
    :param ground_truth: The one-hotted ground_truth as numpy array.
    :return:
    """
    prediction = prediction.round()
    eq = np.all(np.equal(prediction, ground_truth), axis=1)
    return eq.sum() / len(ground_truth)


def calc_confusion_matrix(predicted_labels: [int], ground_truth: [int], class_idx: int) -> dict:
    """Calculates the confusion matrix values (TP, FP, TN, FN).

    :param predicted_labels: The list of predicted labels. (Each label should be an integer between 0 and
                             num_classes - 1, e.g. [0, 1, 4, 2])
    :param ground_truth: The raw ground truth. (E.g. [0, 1, 4, 2])
    :param class_idx: The class index to be examined.
    :return:
    """
    TP = sum(
        1 for i in range(len(predicted_labels)) if predicted_labels[i] == class_idx and ground_truth[i] == class_idx)
    FP = sum(
        1 for i in range(len(predicted_labels)) if predicted_labels[i] == class_idx and ground_truth[i] != class_idx)
    TN = sum(
        1 for i in range(len(predicted_labels)) if predicted_labels[i] != class_idx and ground_truth[i] != class_idx)
    FN = sum(
        1 for i in range(len(predicted_labels)) if predicted_labels[i] != class_idx and ground_truth[i] == class_idx)
    return {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}


def calc_confusion_matrix_2d(prediction: [int], ground_truth: [int], num_classes: int) -> [[int]]:
    """Calculates the confusion matrix for multiple classes.

    :param prediction: The raw list of predicted labels.
    :param ground_truth: The raw ground truth. (E.g. [0, 1, 4, 2])
    :param num_classes: The total number of classes.
    :return: A 2D list representing the confusion matrix.
    """
    predicted_labels = np.argmax(prediction, axis=1)
    confusion_matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for i in range(len(predicted_labels)):
        predicted_label = predicted_labels[i]
        true_label = ground_truth[i]
        confusion_matrix[true_label][predicted_label] += 1
    return confusion_matrix


def calc_precision(conf_mat: dict):
    divisor = (conf_mat["TP"] + conf_mat["FP"])
    divisor = divisor if (conf_mat["TP"] + conf_mat["FP"]) != 0 else 0.00001
    return conf_mat["TP"] / divisor


def calc_recall(conf_mat: dict):
    divisor = (conf_mat["TP"] + conf_mat["FN"])
    return 0 if divisor == 0 else conf_mat["TP"] / divisor


def calc_f1(conf_mat: dict):
    divisor = (2 * (conf_mat["TP"]) + conf_mat["FP"] + conf_mat["FN"])
    return 0 if divisor == 0 else (2 * conf_mat["TP"]) / divisor


def to_onehot(Y, num_classes):
    Y = np.array(Y)
    if num_classes == -1:
        raise AttributeError("The class count is invalid.")
    return np.eye(num_classes)[Y]


def sigmoid(z):
    # Clip the input to a range to avoid overflows
    z_clipped = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z_clipped))


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    # Subtract the maximum value along axis=1 to avoid overflow in the exponential function
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def layers_from_thetas(thetas):
    config = [thetas[0].shape[1] - 1]
    for t in thetas:
        config.append(t.shape[0])
    return config


def create_mini_batches(X, y, batch_size):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    for i in range(0, n_samples, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]
        yield X_batch, y_batch


def is_ffmpeg_installed():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


class Network:
    def __init__(self, layer_shape=(), seed=-1):
        """Initializes a new neural ml_framework.

        :param layer_shape: The shape of the layers. (i.e. [700, 18, 10])
        :param seed: The seed of the ml_framework.
        """
        self.alpha = None
        self.alpha_increase_percent = 0.05
        self.alpha_history = []
        self.ADJUST_ALPHA = False
        self._seed = seed
        self.iterations = None
        if self._seed == -1:
            self._seed = random.randint(0, 2 ** 16)
        np.random.seed(self._seed)

        self._thetas = [np.random.uniform(-1, 1, size=(layer_shape[i], layer_shape[i - 1] + 1)) for i in
                        range(1, len(layer_shape))]
        self.layers = [np.zeros(layer) for layer in layer_shape]
        self._mean = []
        self._std = []
        self._use_regularization = False
        self._lambda_regularization = -1
        self._transformation_matrix = None
        self._class_count = None
        self._class_labels = None
        self.log_intervals = None
        self.batch_size = None

        self._statistical_data = {
            "train": {
                "loss": [],
                "acc": [],
                "precision_per_class": [],
                "recall_per_class": [],
                "f1_per_class": [],
            },
            "val": {
                "loss": [],
                "acc": [],
                "precision_per_class": [],
                "recall_per_class": [],
                "f1_per_class": [],
                "conf_mat_2D": []
            }
        }

    def train(self, hidden_layer_shape, iterations, alpha, X_train, y_train, X_val, y_val, batch_size=None,
              log_intervals=None, class_labels=(), show_plots=False,
              use_feature_scaling=False, lambda_regularization=0.01, use_regularization=False, adjust_alpha=False,
              pca_threshold=0.99):
        self.alpha = alpha
        self.ADJUST_ALPHA = adjust_alpha
        self.iterations = iterations
        self._class_count = np.unique(y_train).size
        self._use_regularization = use_regularization
        self._lambda_regularization = lambda_regularization
        self._set_log_intervals(log_intervals)
        self._set_class_labels(class_labels)
        self.batch_size = batch_size

        if self.batch_size is None:
            self.batch_size = X_train.shape[0]

        if use_feature_scaling:
            # Applying Standardization
            self._mean, self._std = fit(X=X_train)
            X_train_std = standardize(X=X_train, mean=self._mean, std=self._std)
            X_val_std = standardize(X=X_val, mean=self._mean, std=self._std)

            # Applying Principal Component Analysis
            self._transformation_matrix = calc_pca_matrix(X_std=X_train_std,
                                                          threshold=pca_threshold,
                                                          show_plots=show_plots)
            X_train = transform(X=X_train_std, t_mat=self._transformation_matrix)
            X_val = transform(X=X_val_std, t_mat=self._transformation_matrix)

        # Reshape thetas due to principal component analysis
        layer_shape = [X_train.shape[1]]
        layer_shape += hidden_layer_shape
        layer_shape.append(len(class_labels))

        self._thetas = [np.random.uniform(-1, 1, size=(layer_shape[i], layer_shape[i - 1] + 1)) for i in
                        range(1, len(layer_shape))]

        y_train_oh = to_onehot(y_train, self._class_count)
        y_val_oh = to_onehot(y_val, self._class_count)

        X_train = add_bias(X_train)
        X_val = add_bias(X_val)

        # Customize the tqdm progress bar format
        postfix_alpha = f"{self.alpha:.8f}".rstrip('0').rstrip('.') if self.alpha else '0'
        progress_bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [elapsed: {elapsed} remaining: {remaining}] {postfix}"

        with tqdm(range(iterations), bar_format=progress_bar_format, postfix=f"Alpha: {postfix_alpha} ") as pbar:
            for iteration in pbar:
                if len(X_val) > 0:
                    h = self.predict(X_val)
                    self.data_analysis_advanced(dict_dir='val', prediction=h, ground_truth=y_val,
                                                ground_truth_oh=y_val_oh, iteration=iteration)

                # Create mini-batches
                mini_batches = create_mini_batches(X_train, y_train_oh, self.batch_size)

                all_train_predictions = []
                all_train_ground_truth_oh = []

                for X_batch, y_batch in mini_batches:
                    # FORWARD PASS
                    aD = self.predict(X_batch)

                    # Accumulate training predictions and ground truth one-hot encoded labels
                    all_train_predictions.append(aD)
                    all_train_ground_truth_oh.append(y_batch)

                    # BACKWARD PASS
                    dJ_do = aD - y_batch

                    # Compute gradients
                    gradients = self.compute_gradients(self.layers,
                                                       self._thetas,
                                                       dJ_do,
                                                       self._use_regularization,
                                                       self._lambda_regularization)

                    # Update weights
                    self.update_weights(self._thetas, gradients, self.alpha)

                # Concatenate all predictions and ground truth one-hot encoded labels for this iteration
                all_train_predictions = np.vstack(all_train_predictions)
                all_train_ground_truth_oh = np.vstack(all_train_ground_truth_oh)
                self.data_analysis_basic(dict_dir='train', prediction=all_train_predictions,
                                         ground_truth_oh=all_train_ground_truth_oh, iteration=iteration)
                postfix_alpha = f"{self.alpha:.8f}".rstrip('0').rstrip('.') if self.alpha else '0'
                pbar.set_postfix_str(
                    f"Alpha: {postfix_alpha}, Validation Acc: {self._statistical_data['val']['acc'][-1]:.8f}",
                    refresh=True)

        self.report_model_performance()
        self.report_confusion_matrix_2d()

        if show_plots:
            #if is_ffmpeg_installed():
            self.plot_confusion_matrix_2d()
            self.plot_all()
            plt.show()

    @staticmethod
    def update_weights(thetas, gradients, alpha):
        for i, grad in enumerate(gradients):
            thetas[i] = thetas[i] - alpha * grad

    @staticmethod
    def compute_gradients(layers, thetas, dJ_do, use_regularization, lambda_regularization):
        # A -> Z
        dA_z = []
        for i in range(1, len(layers)):
            dA_z.append(layers[i] * (1 - layers[i]))
        # convert to numpy array and reverse, to go from output (index 0) to input layer
        dA_z = dA_z[::-1]

        # J -> Z
        dJ_z = [np.array(dA_z[0] * dJ_do)]
        for i in range(1, len(thetas)):
            res = dA_z[i].T * (thetas[len(thetas) - i][:, 1:].T @ dJ_z[i - 1].T)
            dJ_z.append(np.array(res.T))
        # revert dJ_z to regular front to back order
        dJ_z = dJ_z[::-1]

        # J -> Thetas
        gradients = []
        for i in range(0, len(thetas)):
            layer = layers[i]
            # only insert bias for hidden layers other than input layer X
            if i > 0:
                bias = np.ones((1, layers[i].shape[0]))
                layer = np.insert(layers[i], 0, bias, axis=1)
            grad = np.array(dJ_z[i].T @ layer)
            if use_regularization:
                # L1 Regularization
                grad += lambda_regularization * np.sign(thetas[i])
            gradients.append(grad)
        return gradients

    def predict(self, X, apply_feature_scaling=False):

        # Used for live mode
        if apply_feature_scaling and len(self._mean) > 0:
            X_std = standardize(X=X, mean=self._mean, std=self._std)
            X_pca = transform(X=X_std, t_mat=self._transformation_matrix)
            X = add_bias(X=X_pca)

        # Handle the input layer (no activation applied)
        self.layers[0] = X

        output_index = len(self.layers) - 1

        # Handle the hidden layers (sigmoid applied)
        for i in range(1, len(self.layers) - 1):
            if i == 1:
                prev_layer = self.layers[i - 1]
            else:
                # Add bias
                bias = np.ones((1, self.layers[i - 1].shape[0]))
                prev_layer = np.insert(self.layers[i - 1], 0, bias, axis=1)

            z = self._thetas[i - 1] @ prev_layer.T
            self.layers[i] = sigmoid(z).T

        # Handle the output layer (softmax applied)
        bias = np.ones((1, self.layers[len(self.layers) - 1 - 1].shape[0]))
        prev_layer = np.insert(self.layers[output_index - 1], 0, bias, axis=1)
        z = self._thetas[output_index - 1] @ prev_layer.T
        self.layers[output_index] = softmax(z.T)

        return self.layers[-1]

    @classmethod
    def load(cls, source_dir):
        print(f"Loading existing model from {source_dir}")

        # Load params
        params = dict(np.load(source_dir, allow_pickle=True).items())

        # Retrieve thetas (weights) and reconstruct the _thetas list
        thetas_keys = [key for key in params if key.startswith("theta_")]
        thetas = [params[key] for key in sorted(thetas_keys)]
        params['_thetas'] = thetas

        for key in thetas_keys:  # weights
            del params[key]

        seed = int(params['_seed'])
        del params['_seed']

        network = Network(layer_shape=layers_from_thetas(params['_thetas']), seed=seed)

        for param_name, param_value in params.items():
            setattr(network, param_name, param_value)

        return network

    def save(self, target_dir, timestamp):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # Use inspect to get a list of attribute names
        params = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        hyper_params = [param[0] for param in params if
                        not (param[0].startswith('__') and param[0].endswith('__')) and param[0].startswith('_')]

        # Create a dictionary of attributes to save
        hyper_params_dict = {hyper_param: getattr(self, hyper_param) for hyper_param in hyper_params}

        outfile = f"{target_dir}/hyper_params_{timestamp}.npz"

        thetas_dict = {f"theta_{i}": theta for i, theta in enumerate(hyper_params_dict['_thetas'])}
        del hyper_params_dict['_thetas']

        print(f"Saved to {outfile}")
        combined_dict = {**hyper_params_dict, **thetas_dict}
        np.savez(outfile, **combined_dict)

    def calculate_ce_loss(self, prediction, ground_truth):
        """Calculates the cross entropy loss with L1 regularization penalty for the given prediction and ground truth data.

        :param prediction: A ndarray with shape (n_samples, n_classes) containing the raw prediction values for each sample and class.
        :param ground_truth: A ndarray with shape (n_samples, n_classes) containing the ground truth labels for each sample and class, encoded as one-hot vectors.
        :return: The calculated loss value as a float.
        """
        epsilon = 1e-9
        return -(np.log(prediction + epsilon) * ground_truth).sum() / len(ground_truth) + self.l1_penalty()

    def l1_penalty(self):
        W_sum = 0
        for t in self._thetas:
            W_sum += np.sum(np.abs(t))
        return self._lambda_regularization * W_sum if self._use_regularization else 0

    def evaluate(self, y_pred, ground_truth) -> dict:
        statistical_data = {
            'test': {}
        }
        y_pred_labels = np.argmax(y_pred, axis=1)
        print(self._class_labels)
        f1_p_class, precision_p_class, recall_p_class = calc_per_class_scores(ground_truth=ground_truth,
                                                                              pred_labels=y_pred_labels,
                                                                              class_count=self._class_count)
        test_data = statistical_data['test']
        test_data['recall_per_class'] = recall_p_class
        test_data['precision_per_class'] = precision_p_class
        test_data['f1_per_class'] = f1_p_class
        test_data['conf_mat_2D'] = calc_confusion_matrix_2d(prediction=y_pred,
                                                            ground_truth=ground_truth,
                                                            num_classes=self._class_count)

        y_oh = to_onehot(Y=ground_truth, num_classes=self._class_count)
        test_data['loss'] = self.calculate_ce_loss(y_pred, y_oh)
        test_data['acc'] = calculate_accuracy(y_pred, y_oh)

        self.plot_confusion_matrix_2d(statistical_data["test"]["conf_mat_2D"], labels=self._class_labels)
        return statistical_data

    def data_analysis_basic(self, dict_dir, prediction, ground_truth_oh, iteration):
        """
        Perform basic data analysis on the given prediction and ground truth one-hot encoded data.

        :param dict_dir: A string indicating the directory in statistical_data where the training will be stored.
        :param prediction: A numpy array containing the predicted classes.
        :param ground_truth_oh: A numpy array containing the ground truth one-hot encoded classes.
        :param iteration: An integer representing the current iteration of training.
        """
        if iteration % self.log_intervals['first'] == 0:
            parent_dict = self._statistical_data[dict_dir]
            parent_dict["loss"].append(self.calculate_ce_loss(prediction, ground_truth_oh))
            if self.ADJUST_ALPHA:
                if dict_dir == "train" and len(parent_dict["loss"]) > 1:
                    self.alpha_history.append(self.alpha)
                    if parent_dict["loss"][-1] >= parent_dict["loss"][-2]:
                        self.alpha *= 0.5
                        self.alpha_increase_percent *= 0.5  # slow down alpha increase to 50%
                    else:
                        self.alpha *= (1 + self.alpha_increase_percent)

            parent_dict["acc"].append(calculate_accuracy(prediction, ground_truth_oh))

        if iteration % self.log_intervals['second'] == 0:
            pass

    def data_analysis_advanced(self, dict_dir: str, prediction: np.array, ground_truth: np.array,
                               ground_truth_oh: np.array,
                               iteration: int) -> None:
        """Performs advanced data analysis and updates the statistical data.

        :param dict_dir: The name of the dictionary in the statistical data where the training should be stored.
        :param prediction: A numpy array of shape (batch_size, num_classes) representing the predicted scores for each class.
        :param ground_truth: A numpy array of shape (batch_size,) representing the ground truth labels.
        :param ground_truth_oh: A numpy array of shape (batch_size, num_classes) representing the ground truth one-hot encoded labels.
        :param iteration: An integer representing the current iteration number.
        """
        prediction = np.array(prediction)
        pred_labels = np.argmax(prediction, axis=1)
        ground_truth = np.array(ground_truth)
        parent_dict = self._statistical_data[dict_dir]

        if iteration % self.log_intervals['first'] == 0:
            pass

        if iteration % self.log_intervals['second'] == 0:
            f1_p_class, precision_p_class, recall_p_class = calc_per_class_scores(ground_truth=ground_truth,
                                                                                  pred_labels=pred_labels,
                                                                                  class_count=self._class_count)

            parent_dict["recall_per_class"].append(recall_p_class)
            parent_dict["precision_per_class"].append(precision_p_class)
            parent_dict["f1_per_class"].append(f1_p_class)
            parent_dict["conf_mat_2D"].append(calc_confusion_matrix_2d(prediction=prediction,
                                                                       ground_truth=ground_truth,
                                                                       num_classes=self._class_count))

        if iteration % self.log_intervals['second'] == 0 or iteration == self.iterations - 1:
            pass

        self.data_analysis_basic(dict_dir, prediction, ground_truth_oh, iteration)

    def model_info(self):
        print(">>> Network structure <<<")
        print(f"Input size: {str(len(self.layers[0]))}")
        print(f"Output size: {str(len(self.layers[-1]))}")
        shapes = [len(self.layers[i]) for i in range(1, len(self.layers) - 1)]
        print(f"{len(self.layers) - 2} hidden layers of shapes {str(shapes)}")
        print(f"Total number of neurons: {np.sum(list(map(len, self.layers[1:])))}")
        print(f"Total number of weights: {np.sum(list(map(lambda x: sum(list(map(len, x))), self._thetas)))}")
        print(f"Seed: {self._seed}")
        print(f"Class Count: {self._class_count}")

    def report_confusion_matrix_2d(self):
        conf2D = self._statistical_data['val']['conf_mat_2D'][-1]
        confusion_rates = []
        print("\n=============== Highest Confusion Rate ===============")
        for class_idx in range(self._class_count):
            confused_idx = np.argsort(conf2D[class_idx])[-2]  # Index of 2nd highest value
            confused_percentage = round(conf2D[class_idx][confused_idx] / sum(conf2D[class_idx]) * 100, 2)
            print(f"'{self._class_labels[class_idx]}' mostly confused by "
                  f"'{self._class_labels[confused_idx]}' in {confused_percentage}% of cases")
            confusion_rates.append(confused_percentage)
        print("=================================================================\n")

    def report_model_performance(self) -> None:
        train_acc: float = self._statistical_data["train"]["acc"][-1]
        val_acc: float = self._statistical_data["val"]["acc"][-1]
        recall_per_class: List[float] = self._statistical_data["val"]["recall_per_class"][-1]
        precision_per_class: List[float] = self._statistical_data["val"]["precision_per_class"][-1]
        f1_per_class: List[float] = self._statistical_data["val"]["f1_per_class"][-1]
        print(f"Final training Accuracy: {round(train_acc * 100, 2)}%")
        print(f"Final Validation Accuracy: {round(val_acc * 100, 2)}%")
        print(f"Final Recall Per-Class: {round(np.mean(recall_per_class) * 100, 2)}%")
        print(f"Final Precision Per-Class: {round(np.mean(precision_per_class) * 100, 2)}%")
        print(f"Final F1 Score Per-Class: {round(np.mean(f1_per_class) * 100, 2)}%")

    def plot_confusion_matrix_2d(self, confusion_matrix=None, labels=None):
        """
        This function prints and plots the confusion matrix.
        :param confusion_matrix: confusion matrix (list or ndarray)
        :param labels: list of label names
        """
        if confusion_matrix is None:
            confusion_matrix = self._statistical_data['val']['conf_mat_2D'][-1]
        if labels is None:
            labels = self._class_labels
        plt.subplots(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.title('Confusion Matrix')
        plt.tight_layout()

    def create_confusion_matrix_2d_live_mp4(self, target_dir):
        if not config.PLOT_CONF_MAT_2D:
            return

        if not is_ffmpeg_installed():
            return

        data = self._statistical_data['val']['conf_mat_2D']
        grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
        labels = self._class_labels
        fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw=grid_kws, figsize=(8, 6))
        ax.set_title(f"Iteration 0", fontsize=14)

        def my_func(i):
            ax.cla()
            ax.set_title(f"Iteration {i * self.log_intervals['second']}", fontsize=14)
            sns.heatmap(data[i], ax=ax, cbar=True, cbar_ax=cbar_ax, fmt='d', annot=True, linewidth=.5,
                        xticklabels=labels, yticklabels=labels)
            ax.set_xticklabels(labels, rotation=45)  # Rotate x-axis labels for better visibility.
            fig.subplots_adjust(bottom=0.2)  # Adjust the bottom margin.

        anim = FuncAnimation(fig=fig, func=my_func, frames=len(data), interval=200, blit=False)
        anim.save(target_dir, writer='ffmpeg', fps=10, dpi=300)

    def plot_all(self):
        """Plots all four plots in a 2x2 grid."""
        fig, axs = plt.subplots(2, 2, figsize=(17, 16))
        self.plot_loss_and_acc(axs[0, 0])
        self.plot_precision(axs[0, 1])
        self.plot_recall(axs[1, 0])
        self.plot_f1_score(axs[1, 1])
        fig.subplots_adjust(hspace=0.5)  # Add some vertical spacing between rows

    def plot_loss_and_acc(self, axs):
        """Plots the training and validation loss and accuracy over iterations."""
        train = self._statistical_data["train"]
        val = self._statistical_data["val"]

        axs.set_title("Total Accuracy and Loss", fontsize=14)
        axs.plot(np.repeat(train["loss"], self.log_intervals['first']), color="red", label="training Loss")
        axs.plot(np.repeat(val["loss"], self.log_intervals['first']), color="orange", label="Validation Loss")
        axs.legend(loc='lower left')
        axs.set_ylabel("Absolute Loss", color="red", fontsize=12)
        axs.set_xlabel("Iterations", fontsize=12)

        ax1b = axs.twinx()
        ax1b.plot(np.repeat(train["acc"], self.log_intervals['first']), color="blue", label="training Accuracy")
        ax1b.plot(np.repeat(val["acc"], self.log_intervals['first']), color="green", label="Validation Accuracy")
        ax1b.set_ylabel("Accuracy in %", color="blue", fontsize=12)
        ax1b.legend(loc='upper left')

    def plot_precision(self, axs):
        """Plots the precision per class."""
        precision_p_class = np.array(self._statistical_data["val"]["precision_per_class"])

        axs.set_title("Precision per Class", fontsize=14)
        for i in range(precision_p_class.shape[1]):
            axs.plot(np.repeat(precision_p_class[:, i], self.log_intervals['second']), label=self._class_labels[i])
        axs.set_ylabel("Precision in %", color="orange", fontsize=12)
        axs.set_xlabel("Iterations", fontsize=12)
        axs.yaxis.tick_right()
        axs.yaxis.set_label_position('right')
        axs.legend(loc="best")

    def plot_recall(self, axs):
        """Plots the recall per class."""
        recall_p_class = np.array(self._statistical_data["val"]["recall_per_class"])

        for i in range(recall_p_class.shape[1]):
            axs.plot(np.repeat(recall_p_class[:, i], self.log_intervals['second']), label=self._class_labels[i])
        axs.set_ylabel("Recall in %", color="orange", fontsize=12)
        axs.set_xlabel("Iterations", fontsize=12)
        axs.set_title("Recall per Class", fontsize=14)
        axs.yaxis.tick_right()
        axs.yaxis.set_label_position('right')
        axs.legend(loc="best")

    def plot_f1_score(self, axs):
        """ Plots the F1 score for each class over the training iterations."""
        f1_p_class = np.array(self._statistical_data["val"]["f1_per_class"])

        for i in range(f1_p_class.shape[1]):
            axs.plot(np.repeat(f1_p_class[:, i], self.log_intervals['second']), label=self._class_labels[i])
        axs.set_ylabel("F1 Score in %", color="green", fontsize=12)
        axs.set_xlabel("Iterations", fontsize=12)
        axs.set_title("F1 Score per Class", fontsize=14)
        axs.yaxis.tick_right()
        axs.yaxis.set_label_position('right')
        axs.legend(loc="best")

    def _set_log_intervals(self, log_intervals):
        if log_intervals is None:
            self.log_intervals = {'first': 1, 'second': 10}
        else:
            self.log_intervals = log_intervals

    def _set_class_labels(self, class_labels):
        class_labels_len = len(class_labels)
        if class_labels_len != 0 and class_labels_len != self._class_count:
            raise AttributeError(
                f"The amount of given class labels ({class_labels_len}) does not match class count in the given "
                f"training dataset ({self._class_count}). Consider to either choose a training dataset with the matching "
                f"class amount, or regenerate your training dataset.")
        if class_labels_len == 0:
            self._class_labels = [i for i in range(self._class_count)]
        else:
            self._class_labels = class_labels

    def get_class_labels(self):
        return self._class_labels
