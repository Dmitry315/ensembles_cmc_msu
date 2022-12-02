from time import time
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor


class RandomForestMSE:
    def __init__(
            self, n_estimators, max_depth=None, feature_subsample_size=None,
            **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        if feature_subsample_size is None:
            self.feature_subsample_size = 1 / 3
        self.trees_params = trees_parameters
        # help params
        self.val_rmse_history = []
        self.train_rmse_history = []
        self.dt = []
        self.trees = []

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features
        y_val : numpy ndarray
            Array of size n_val_objects
        """
        if X_val is not None:
            self.val_rmse_history = []
        self.train_rmse_history = []
        self.dt = []
        t0 = time()
        n_objects = X.shape[0]
        n_subset = int((1 - 1 / np.e) * n_objects)
        self.trees = []
        if X_val is not None:
            pred_val = np.zeros_like(y_val)
        pred_train = np.zeros_like(y)
        for i in range(1, self.n_estimators + 1):
            tree = DecisionTreeRegressor(max_depth=self.max_depth, max_features=self.feature_subsample_size,
                                         **self.trees_params)
            # random patch
            idx = np.arange(n_objects)
            np.random.shuffle(idx)
            sub_idx = np.isin(np.arange(n_objects), idx[:n_subset])

            # train model
            tree.fit(X[sub_idx], y[sub_idx])
            self.dt.append(time() - t0)
            # val score
            if X_val is not None:
                if i == 1:
                    pred_val += tree.predict(X_val)
                else:
                    pred_val *= (i - 1) / i
                    pred_val += tree.predict(X_val) / i
                self.val_rmse_history.append(
                    np.sqrt(np.mean(np.square(pred_val - y_val)))
                )
            # train score
            if i == 0:
                pred_train += tree.predict(X)
            else:
                pred_train *= (i - 1) / i
                pred_train += tree.predict(X) / i
            self.train_rmse_history.append(
                np.sqrt(np.mean(np.square(pred_train - y)))
            )

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        pred = np.zeros(X.shape[1])
        for tree in self.trees:
            pred += tree.predict(X)
        return pred / len(self.trees)


class GradientBoostingMSE:
    def __init__(
            self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
            **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        learning_rate : float
            Use alpha * learning_rate instead of alpha
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        if feature_subsample_size is None:
            self.feature_subsample_size = 1 / 3
        self.trees_params = trees_parameters
        # help params
        self.val_rmse_history = []
        self.train_rmse_history = []
        self.dt = []
        self.trees = []
        self.w = []

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        """
        if X_val is not None:
            self.val_rmse_history = []
        self.train_rmse_history = []
        self.dt = []
        t0 = time()

        self.trees = []
        self.w = []
        z = np.zeros_like(y)
        if X_val is not None:
            pred_val = np.zeros_like(y_val)
        pred_train = np.zeros_like(y)
        for i in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth, max_features=self.feature_subsample_size,
                                         **self.trees_params)
            if i == 0:  # initial approximation
                tree.fit(X, y)
                self.dt.append(time() - t0)
                z += tree.predict(X)
                self.trees.append(tree)
                self.w.append(1)
                # val score
                if X_val is not None:
                    pred_val += self.w[-1] * tree.predict(X_val)

                    self.val_rmse_history.append(
                        np.sqrt(np.mean(np.square(pred_val - y_val)))
                    )
                # train score
                pred_train += self.w[-1] * tree.predict(X)

                self.train_rmse_history.append(
                    np.sqrt(np.mean(np.square(pred_train - y)))
                )
                continue
            s = z - y  # grad_z mse(y, z)
            tree.fit(X, s)
            self.dt.append(time() - t0)
            preds = tree.predict(X)
            res = minimize_scalar(lambda w, z, preds: np.mean(np.square(z + w * preds - y)), args=(z, preds))
            w = self.learning_rate * res.x
            z += w * preds
            self.w.append(w)
            self.trees.append(tree)
            # val score
            if X_val is not None:
                pred_val += self.w[-1] * tree.predict(X_val)

                self.val_rmse_history.append(
                    np.sqrt(np.mean(np.square(pred_val - y_val)))
                )
            # train score
            pred_train += self.w[-1] * tree.predict(X)

            self.train_rmse_history.append(
                np.sqrt(np.mean(np.square(pred_train - y)))
            )

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        pred = np.zeros(X.shape[1])
        for tree, w in zip(self.trees, self.w):
            pred += tree.predict(X) * w
        return pred
