import numpy as np
from src import util
from src.linear_model import LinearModel

# import util
# from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)

    # *** START CODE HERE ***
    clf = GDA()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_eval)
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m = x.shape[0]
        phi = 1 / m * np.sum(y)
        mu_0 = x.T @ (1 - y) / np.sum(1 - y)
        mu_1 = x.T @ y / np.sum(y)
        sigma = (
            1
            / m
            * (
                (x - mu_0).T @ ((1 - y) * np.identity(m)) @ (x - mu_0)
                + (x - mu_1).T @ (y * np.identity(m)) @ (x - mu_1)
            )
        )

        self.theta = np.linalg.inv(sigma) @ (mu_1 - mu_0)
        self.theta_0 = 1 / 2 * (mu_0 + mu_1).T @ np.linalg.inv(sigma) @ (mu_0 - mu_1) - np.log(
            (1 - phi) / phi
        )
        self.theta = np.insert(self.theta, 0, self.theta_0)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return x @ self.theta >= 0
        # *** END CODE HERE
