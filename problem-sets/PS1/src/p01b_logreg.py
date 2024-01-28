import numpy as np
import util
from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    clf.predict(x_eval)
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

        return -1 / x.shape[0] * x.T @ (y - self.h(theta, x))

    def hessian(self, theta, x):
        h_theta_x = np.reshape(self.h(theta, x), (-1, 1))
        return 1 / x.shape[0] * x.T @ (h_theta_x * (1 - h_theta_x) * x)

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        self.initialize_theta(x.shape[1])

        for _ in range(10):
            self.theta = self.theta - np.linalg.inv(self.hessian(self.theta, x)) @ self.gradient(
                self.theta, x, y
            )

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
        return self.h(self.theta, x)
        # *** END CODE HERE ***
