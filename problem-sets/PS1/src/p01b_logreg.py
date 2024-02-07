import numpy as np
from src import util
from src.linear_model import LinearModel

# import util
# from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    print(pred_path)

    # *** START CODE HERE ***
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_eval)
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def h(self, theta, x):
        """Prediction of logistic regression

        Args:
            theta: Parameters with shape (n,).
            x: Single example with shape (m, n).
            output: Prediction of logistic regression with shape (m,).
        """

        return 1 / (1 + np.exp(-x @ theta))

    def gradient(self, theta, x, y):
        """Gradient of l(theta) which is a log-likelihood function

        Args:
            theta: Parameters with shape (n,).
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        m = x.shape[0]
        return 1 / m * x.T @ (self.h(theta, x) - y)

    def hessian(self, theta, x):
        h_theta_x = np.reshape(self.h(theta, x), (-1, 1))
        m = x.shape[0]
        return 1 / m * x.T @ (h_theta_x * (1 - h_theta_x) * np.identity(m)) @ x

    def loss_function(self, theta, x, y):
        h_theta_x = self.h(theta, x)
        m = x.shape[0]

        return -1 / m * y @ np.log(h_theta_x) + (1 - y) @ np.log(h_theta_x)

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        num_features = x.shape[1]
        self.initialize_theta(num_features)
        epsilon = 1e-5

        while True:
            self.theta = self.theta - np.linalg.inv(self.hessian(self.theta, x)) @ self.gradient(
                self.theta, x, y
            )
            if self.loss_function(self.theta, x, y) < epsilon:
                break

        # *** END CODE HERE ***

    def initialize_theta(self, n):
        self.theta = np.zeros(n)

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return self.h(self.theta, x) >= 0.5
        # *** END CODE HERE ***
