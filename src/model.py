import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from src.features import NUMERIC, CATEGORICAL


def make_model(kind, seed):
    numeric = [('impute', SimpleImputer(strategy='median', keep_empty_features=True))]
    if kind == 'logistic':
        numeric.append(('scale', StandardScaler()))
    preprocess = ColumnTransformer([
        ('numeric', Pipeline(numeric), NUMERIC),
        ('codes', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=10,
                               max_categories=35, sparse_output=False), CATEGORICAL)])
    # Modest cost-sensitive weights, rather than inverse-frequency weights or SMOTE.
    weights = {0: 1.0, 1: 2.0, 2: 4.0}
    if kind == 'logistic':
        estimator = LogisticRegression(max_iter=1500, C=.5, class_weight=weights, random_state=seed)
    else:
        estimator = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15,
                    min_samples_leaf=25, l2_regularization=2, learning_rate=.06,
                    early_stopping=False, class_weight=weights, random_state=seed)
    return Pipeline([('preprocess', preprocess), ('classifier', estimator)])


class CalibratedModel:
    """Temperature scaling on a separate temporal set with natural class balance."""
    def __init__(self, model):
        self.model = model
        self.temperature = 1.0

    def fit_calibration(self, X, y):
        logits = np.log(np.clip(self.model.predict_proba(X), 1e-9, 1))
        result = minimize_scalar(lambda t: log_loss(y, softmax(logits/t, axis=1), labels=[0,1,2]),
                                 bounds=(.35, 4), method='bounded')
        self.temperature = float(result.x)
        return self

    def predict_proba(self, X):
        logits = np.log(np.clip(self.model.predict_proba(X), 1e-9, 1))
        return softmax(logits/self.temperature, axis=1)
