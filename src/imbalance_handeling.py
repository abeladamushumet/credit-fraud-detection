import pandas as pd
from typing import Tuple, Optional, Union
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_data(
    df: pd.DataFrame,
    target_column: str = 'is_fraud',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into training and testing sets.
    """
    logger.info("Splitting data...")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: Union[str, float] = 'auto',
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies SMOTE to oversample the minority class.
    """
    logger.info("Applying SMOTE oversampling...")
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE: {y_res.value_counts().to_dict()}")
    return X_res, y_res


def apply_undersampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: Union[str, float] = 'auto',
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies random undersampling to reduce the majority class.
    """
    logger.info("Applying Random Undersampling...")
    undersample = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = undersample.fit_resample(X_train, y_train)
    logger.info(f"After Undersampling: {y_res.value_counts().to_dict()}")
    return X_res, y_res


def apply_combined_sampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    over_strategy: float = 0.1,
    under_strategy: float = 0.5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies SMOTE followed by Random Undersampling.
    """
    logger.info(f"Applying combined SMOTE + Undersampling (over={over_strategy}, under={under_strategy})...")
    over = SMOTE(sampling_strategy=over_strategy, random_state=random_state)
    under = RandomUnderSampler(sampling_strategy=under_strategy, random_state=random_state)
    pipeline = Pipeline(steps=[('smote', over), ('under', under)])
    X_res, y_res = pipeline.fit_resample(X_train, y_train)
    logger.info(f"After Combined Sampling: {y_res.value_counts().to_dict()}")
    return X_res, y_res


def balance_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = 'smote',
    over_strategy: float = 0.1,
    under_strategy: float = 0.5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    High-level function to choose and apply a balancing method.

    method options:
    - 'smote'
    - 'undersample'
    - 'combined'
    """
    method = method.lower()
    logger.info(f"Selected balancing method: {method}")

    if method == 'smote':
        return apply_smote(X_train, y_train, sampling_strategy=over_strategy, random_state=random_state)
    elif method == 'undersample':
        return apply_undersampling(X_train, y_train, sampling_strategy=under_strategy, random_state=random_state)
    elif method == 'combined':
        return apply_combined_sampling(X_train, y_train, over_strategy, under_strategy, random_state)
    else:
        raise ValueError("Unsupported balancing method. Choose from 'smote', 'undersample', or 'combined'.")
