import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, recall_score
from GNB import GaussianNaiveBayes

def cv_score_var_smoothing(
    X, y, var_smoothing,
    n_splits=5,
    scoring="f1_weighted",
    random_state=42
):
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = GaussianNaiveBayes(var_smoothing=var_smoothing)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        if scoring == "accuracy":
            score = accuracy_score(y_val, y_pred)
        elif scoring == "f1_weighted":
            score = f1_score(y_val, y_pred, average="weighted", zero_division=0)
        elif scoring == "recall":
            score = recall_score(y_val, y_pred, average="weighted", zero_division=0)
        else:
            raise ValueError("Unsupported scoring")

        scores.append(score)

    return np.mean(scores), np.std(scores)
