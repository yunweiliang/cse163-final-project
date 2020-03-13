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
        self._y = self._data['prediction']
        self.setX()


    def setX(self, x = None):
        if x is None:
            self._X = self._data.loc[:, self._data.columns != 'prediction']
            self._X = pd.get_dummies(self._X)
        else:
            self._X = x

    def decision_tree(self):
        X_train, X_test, y_train, y_test = train_test_split(self._X, self._y, test_size=0.2)
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        result = accuracy_score(y_test, model.predict(X_test))

        return result

    def naive_bayes(self):
        X_train, X_test, y_train, y_test = train_test_split(self._X, self._y, test_size=0.2)
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        score_test = accuracy_score(y_test, y_test_pred)
        return score_test

    def forest(self):
        data =  self._data.drop('prediction', axis=1)
        col_name = list(data.columns)

        X_train, X_test, y_train, y_test = train_test_split(self._X, self._y, test_size=0.2)
        rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)
        rf_model.fit(X_train, y_train)

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