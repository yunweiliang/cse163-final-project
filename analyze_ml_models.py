import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class ML_Model:
    def __init__(self, file_path):
        self._data = pd.read_csv(file_path)
        self._data = self._data.dropna().reset_index(drop=True)
        # Make into binary problem
        self._data.loc[self._data['prediction'] > 0, 'prediction'] = 1
        self._feature_importances = None

    def decision_tree(self):
        X = self._data.loc[:, self._data.columns != 'prediction']
        X = pd.get_dummies(X)
        y = self._data['prediction']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        result = accuracy_score(y_test, model.predict(X_test))
        return result

    def naive_bayes(self):
        X = self._data.loc[:, self._data.columns != 'prediction']
        X = pd.get_dummies(X)
        y = self._data['prediction']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = GaussianNB()
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        #score_train = accuracy_score(y_train, y_train_pred)
        score_test = accuracy_score(y_test, y_test_pred)
        return score_test

    def forest(self):
        np_label  =  np.array(self._data['prediction'])
        data =  self._data.drop('prediction', axis=1)
        col_name = list(data.columns)
        np_data = np.array(data)
        # split data
        X_train, X_test, y_train, y_test = train_test_split(np_data, np_label, test_size=0.2)
        # Train model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        #print(rf_model.score(X_train, y_train))

        # save feature importances for Part 2: feature selection
        importances = list(rf_model.feature_importances_)
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(col_name, importances)]
        self._feature_importances = sorted(feature_importances, key=lambda x:x[1], reverse=True)
        #[print('Variable: {} Importance: {}'.format(*pair)) for pair in feature_importances]

        return rf_model.score(X_test, y_test)


def main():
    model = ML_Model('cleveland_processed.csv')

    print('Decision Tree Score:', model.decision_tree())
    print('Gaussian Naive Bayes Score:', model.naive_bayes())
    print('Random Forest Score:', model.forest())

if __name__ == '__main__':
    main()