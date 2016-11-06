import urllib.request
import pandas
import xgboost
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# URL for the Iris dataset (UCI Machine Learning Repository)
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# download the file
raw_data = urllib.request.urlopen(url)

# load the CSV file as a numpy matrix
data = pandas.read_csv(raw_data, header=None)
dataset = data.values

# split data into X and y
X = dataset[:, 0:4]
Y = dataset[:, 4]

# encode string class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
label_encoded_y = label_encoder.transform(Y)

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,
																	label_encoded_y,
																	test_size=test_size,
																	random_state=seed)

# fit model no training data

model = xgboost.XGBClassifier()
model.fit(X_train, y_train)

print(model)

# make the predictions on the test data

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# determine the accuracy of the classifer

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))