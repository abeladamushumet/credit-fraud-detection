import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


def split_data(df, target_column='is_fraud', test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def apply_smote(X_train, y_train, sampling_strategy='auto', random_state=42):
    """
    Applies SMOTE oversampling to balance the minority class.
    """
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


def apply_undersampling(X_train, y_train, sampling_strategy='auto'):
    """
    Applies random undersampling to balance the majority class.
    """
    undersample = RandomUnderSampler(sampling_strategy=sampling_strategy)
    X_res, y_res = undersample.fit_resample(X_train, y_train)
    return X_res, y_res


def apply_combined_sampling(X_train, y_train, over_strategy=0.1, under_strategy=0.5, random_state=42):
    """
    Applies a combination of SMOTE and random undersampling using a pipeline.
    """
    over = SMOTE(sampling_strategy=over_strategy, random_state=random_state)
    under = RandomUnderSampler(sampling_strategy=under_strategy)

    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)

    X_res, y_res = pipeline.fit_resample(X_train, y_train)
    return X_res, y_res
