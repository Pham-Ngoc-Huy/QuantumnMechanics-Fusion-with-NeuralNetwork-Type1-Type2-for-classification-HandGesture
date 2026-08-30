import numpy as np
from scipy.io import loadmat


class GestureClassifier:
    def __init__(self, model_path):
        model = loadmat(model_path, simplify_cells=True)["gesture_model"]

        self.w_hidden = model["w_hidden"]
        self.w_out = model["w_out"]

        self.x_min = model["X_min"]
        self.x_max = model["X_max"]

        self.classes = model["classes"]

    @staticmethod
    def bipolar_sigmoid(x):
        """
        MATLAB:
            bipolar_sigmoid(x)

        f(x) = 2/(1+exp(-x)) - 1
        """
        return 2.0 / (1.0 + np.exp(-x)) - 1.0

    def forward(self, x):
        """
        Parameters
        ----------
        x : ndarray (num_features,)
            One feature vector.

        Returns
        -------
        probabilities : ndarray (num_classes,)
        """

        # convert to column batch
        current = x.reshape(-1, 1)

        for w in self.w_hidden:
            bias = np.ones((1, current.shape[1]))

            u_bias = np.vstack([bias, current])

            net = w.T @ u_bias

            current = self.bipolar_sigmoid(net)

        bias = np.ones((1, current.shape[1]))

        u_bias = np.vstack([bias, current])

        net = self.w_out.T @ u_bias

        # stable softmax
        net -= np.max(net, axis=0, keepdims=True)

        exp = np.exp(net)

        prob = exp / np.sum(exp, axis=0, keepdims=True)

        return prob.squeeze()

    def predict(self, feature_vector):
        x = (feature_vector - self.x_min) / (self.x_max - self.x_min + 1e-8)

        prob = self.forward(x)

        idx = np.argmax(prob)

        return (
            int(self.classes[idx]),
            float(prob[idx]),
        )
