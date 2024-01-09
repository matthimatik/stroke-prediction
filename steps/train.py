from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def estimator_fn(estimator_params: Dict[str, Any] = None):
    pipe = Pipeline([("classifier", RandomForestClassifier())])

    param_grid = [
        {
            "classifier": [RandomForestClassifier(class_weight="balanced")],
            "classifier__n_estimators": [10, 50, 100],
            "classifier__max_depth": [None, 5, 10],
        },
        # {
        #     "classifier": [SVC(class_weight="balanced")],
        #     "classifier__C": [1, 10, 100],
        #     "classifier__kernel": ["linear", "rbf"],
        # },
    ]

    # Create a GridSearchCV object
    return LogisticRegression(class_weight='balanced')
    
    grid_search = GridSearchCV(pipe, param_grid, cv=5, verbose=0, n_jobs=-1)

    return grid_search
