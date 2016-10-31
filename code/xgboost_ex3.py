import numpy as np
from sklearn.model_selection import KFold

X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

kf = KFold(n_splits=10)
for train, test in kf.split(X):
	print("%s %s" % (train, test))
