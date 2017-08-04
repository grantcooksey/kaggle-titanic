import numpy as np
import pandas as pd
import datetime

SUBMISSION_LABELS = ['PassengerId', 'Survived']


def save_submission(prediction, labels, name=None):
    if name is None:
        name='../submissions/' + str(datetime.datetime.now()).split('.')[0].replace(' ', 't') + '.csv'
    result = pd.DataFrame([labels, prediction]).transpose().astype(int)
    result.columns = SUBMISSION_LABELS
    result.to_csv(name, index=False)
