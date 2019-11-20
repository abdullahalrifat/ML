import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

# load data
train_data = pd.read_csv("titanic/train.csv")
test_data = pd.read_csv("titanic/test.csv")

features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

y = train_data['Survived']

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Random forest prediction
forest_model = RandomForestClassifier(n_estimators=100)
forest_model.fit(train_X, train_y)
forest_val_predictions = forest_model.predict(val_X)
print('MAE of Random Forests: %f' % (mean_absolute_error(val_y, forest_val_predictions)))

# Decision Tree prediction
tree_model = DecisionTreeRegressor(random_state=1)
tree_model.fit(train_X, train_y)
tree_val_predictions = tree_model.predict(val_X)
print('MAE of Decision Tree: %f' % (mean_absolute_error(val_y, tree_val_predictions)))

# Knn
knn_model = KNeighborsClassifier(n_neighbors=4)
knn_model.fit(train_X, train_y)
knn_model_predictions = knn_model.predict(val_X)
print('MAE of Knn Model: %f' % (mean_absolute_error(val_y, knn_model_predictions)))

# Neural
mlp_model = MLPClassifier(hidden_layer_sizes=(8,8,8,8), activation='relu', solver='adam', max_iter=50000)
mlp_model.fit(train_X, train_y)
mlp_model_predictions = mlp_model.predict(val_X)
print('MAE of Neural Model: %f' % (mean_absolute_error(val_y, mlp_model_predictions)))

# svm
svclassifier_model = SVC(kernel='rbf')
svclassifier_model.fit(train_X, train_y)
svclassifier_model_predictions = svclassifier_model.predict(val_X)
print('MAE of SVC Model: %f' % (mean_absolute_error(val_y, svclassifier_model_predictions)))

# svm
ada_boost_classifier_model = AdaBoostClassifier(n_estimators=100, random_state=0)
ada_boost_classifier_model.fit(train_X, train_y)
ada_boost_classifier_model_predictions = ada_boost_classifier_model.predict(val_X)
print('MAE of Ada Boost Model: %f' % (mean_absolute_error(val_y, ada_boost_classifier_model_predictions)))


# testing
X_test = pd.get_dummies(test_data[features])
predictions = mlp_model.predict(X_test)

# generate CSV
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print('csv generated.')
