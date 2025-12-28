from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

def tune_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=1   # IMPORTANT: avoid freezing
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt"]
    }

    search = RandomizedSearchCV(
        rf,
        param_distributions=param_grid,
        n_iter=5,          # SMALL but valid
        cv=3,              # faster than 5
        scoring="f1",
        random_state=42,
        verbose=1,
        n_jobs=1           # CRITICAL
    )

    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_


 