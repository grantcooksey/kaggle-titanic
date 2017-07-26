import numpy as np
import pandas as pd

def simple_feature_processing(train, test):
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
