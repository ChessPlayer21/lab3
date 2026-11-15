from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    X = np.asarray(data.train_features, dtype=float)
    y = np.asarray(data.train_target, dtype=float)
    n, p = X.shape
    lr = data.learning_rate

    w = np.zeros(p)
    b = 0.0
    history = []

    for _ in range(data.epochs):
        y_pred = X @ w + b
        err = y_pred - y
        dw = (2 / n) * (X.T @ err)
        db = (2 / n) * err.sum()
        w -= lr * dw
        b -= lr * db
        history.append(float(np.mean((X @ w + b - y) ** 2)))

    preds = np.asarray(data.test_features, dtype=float) @ w + b
    return RegressionResult(weights=w, bias=b, loss_history=history, predictions=preds)



def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    X = np.asarray(data.train_features, dtype=float)
    Y = np.asarray(data.train_target, dtype=float)
    lr = data.learning_rate
    epochs = data.epochs
    n, p = X.shape

    weights = np.zeros(p)
    bias = 0.0
    loss_history = []

    for epoch in range(epochs):
        z = X @ weights + bias
        prob = 1 / (1 + np.exp(-z))
        error = prob - Y

        dw = (1 / n) * (X.T @ error)
        db = (1 / n) * error.sum()

        weights = weights - lr * dw
        bias = bias - lr * db

        eps = 1e-15
        prob_clipped = np.clip(prob, eps, 1 - eps)
        bce = -np.mean(Y * np.log(prob_clipped) + (1 - Y) * np.log(1 - prob_clipped))
        loss_history.append(bce)

    X_test = np.asarray(data.test_features, dtype=float)
    z_test = X_test @ weights + bias
    probabilities = 1 / (1 + np.exp(-z_test))
    predictions = (probabilities >= 0.5).astype(int)

    return ClassificationResult(
        weights=weights,
        bias=bias,
        loss_history=loss_history,
        predictions=predictions,
        probabilities=probabilities,
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    Xtr, ytr = np.asarray(data.train_features, float), np.asarray(data.train_target, int)
    Xte, yte = np.asarray(data.test_features, float), np.asarray(data.test_target, int)

    own = train_logistic_classifier(data)
    tree = DecisionTreeClassifier(random_state=42).fit(Xtr, ytr)
    sc = StandardScaler().fit(Xtr)
    knn = KNeighborsClassifier().fit(sc.transform(Xtr), ytr)

    preds = {
        "own": (own.predictions, own.probabilities),
        "tree": (tree.predict(Xte), tree.predict_proba(Xte)[:, 1]),
        "knn": (knn.predict(sc.transform(Xte)), knn.predict_proba(sc.transform(Xte))[:, 1]),
    }

    metrics = {
        k: ClassificationMetrics(
            roc_auc=roc_auc_score(yte, pr),
            accuracy=accuracy_score(yte, p),
            precision=precision_score(yte, p, zero_division=0),
            recall=recall_score(yte, p, zero_division=0),
            f1=f1_score(yte, p, zero_division=0),
        )
        for k, (p, pr) in preds.items()
    }

    return BaselineComparisonResult(
        logistic_regression=metrics["own"],
        decision_tree=metrics["tree"],
        knn=metrics["knn"],
    )
