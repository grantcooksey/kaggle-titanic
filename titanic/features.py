import numpy as np
import pandas as pd
from sklearn import preprocessing


def logistic_regression_feature_processing(train, test):
    """Preprocessing for the quick and dirty model"""
    age = train['Age']
    train['Age'] = train['Age'].fillna(age.median())
    test['Age'] = test['Age'].fillna(age.median())
    
    fare = train['Fare']
    test['Fare'] = test['Fare'].fillna(fare.median())
    
    feature_labels_reduced = ['Pclass', 'SibSp', 'Parch', 'Sex', 'Embarked']
    feature_labels = feature_labels_reduced + ['Fare', 'Age']
    output_label = 'Survived'
    
    target = train[output_label]
    train = train[feature_labels]
    test = test[feature_labels]
    
    train_matrix =  pd.get_dummies(train, columns=feature_labels_reduced, dummy_na=True)
    test_matrix = pd.get_dummies(test, columns=feature_labels_reduced, dummy_na=True)
    
    # Hack
    train_matrix.insert(21, 'Parch_9.0', 0)
    
    return train_matrix, test_matrix, target


def decision_tree_preprocessing(data, target, categorical_features=None, numerical_features=None, drop_na_columns=None):
    if categorical_features is None:
        categorical_features = []
    if numerical_features is None:
        numerical_features = []
    if drop_na_columns is None:
        drop_na_columns = []

    full_features = categorical_features + numerical_features
    df = data[full_features]
    y = data[target]

    # Drop columns with nulls
    for col in drop_na_columns:
        df_free = df.dropna(subset=[col])
        only_na = df[~df.index.isin(df_free.index)]
        df = df_free
        y = y.drop(only_na.index)

    # Impute data using median
    imp = preprocessing.Imputer(missing_values='NaN', strategy='median', axis=0)
    df[numerical_features] = imp.fit_transform(df[numerical_features])

    # Transform categorical data into numerical labels
    df[categorical_features] = df[categorical_features].apply(preprocessing.LabelEncoder().fit_transform)

    return df, y
