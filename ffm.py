import pandas as pd
import numpy as np
import xlearn as xl
import csv
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc 

moviesDF = pd.read_csv('movielens/user_ratedmovies.dat', header=0, delimiter='\t', usecols=['userID', 'movieID', 'rating'], nrows=10000)
moviesDF['rating'] = moviesDF['rating'] > moviesDF['rating'].mean()

y = np.where(moviesDF['rating'].values, 1, 0)

xUserId = np.array(['0:{}:1'.format(i) for i in moviesDF['userID'].values])
xMovieId = np.array(['1:{}:1'.format(i) for i in moviesDF['movieID'].values])
X = np.stack((xUserId, xMovieId), axis=-1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

X_train_transposed = X_train.T
train_output = np.stack((y_train, X_train_transposed[0], X_train_transposed[1]), axis=-1)

with open('tmp/train.dat', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(train_output)

X_val_transposed = X_val.T
test_output = np.stack((y_val, X_val_transposed[0], X_val_transposed[1]), axis=-1)

with open('tmp/test.dat', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(test_output)

model = xl.create_ffm()
model.setTrain('tmp/train.dat')
model.setValidate('tmp/test.dat')
model.setTest('tmp/test.dat')

param = {
    'task': 'binary',
    'lr': 0.2,
    'lambda': 0.002,
    'metric': 'acc'
}
model.fit(param, 'tmp/model.dat')

model.setSigmoid()
model.predict('tmp/model.dat', 'tmp/preds.dat')

with open('tmp/preds.dat', 'r') as f:
    y_pred = np.array([float(i) for i in f.readlines()])
    fpr, tpr, _ = roc_curve(y_val, y_pred)
    roc_auc = auc(fpr, tpr)
    print('FFM AUC: {}'.format(roc_auc))