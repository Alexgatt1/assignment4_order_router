from __future__ import annotations

from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

feature_columns: List[str] = [
    "side_is_buy",
    "orderqty",
    "limitprice",
    "bid_price",
    "ask_price",
    "bid_size",
    "ask_size",
]

target_column = "price_improvement"
exchange_column = "lastmkt"
min_rows_per_exchange = 200

def load_training_dataframe(path: str = "executions_with_nbbo.parquet") -> pd.DataFrame:
    """
    Load the engineered dataset and prepare it for model training.

    Converts Side ('1' = buy) into a numeric side_is_buy feature and drops
    rows missing any required fields.
    """
    dataframe = pd.read_parquet(path)
    dataframe.columns = [col.lower() for col in dataframe.columns]
    dataframe["side"] = dataframe["side"].astype(str)
    dataframe["side_is_buy"] = (dataframe["side"] == "1").astype(int)
    dataframe = dataframe.dropna(
        subset=feature_columns + [target_column, exchange_column]
    )

    return dataframe
dataframe.head()

def build_grid_search_model() -> GridSearchCV:
    """
    Create a RandomForest regression pipeline wrapped in GridSearchCV.

    The grid search explores a small parameter space and uses n_jobs=-1
    for full parallelism.
    """
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
        ]
    )

    param_grid = {
        "model__n_estimators": [50, 100],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_leaf": [1, 5],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    return grid_search
