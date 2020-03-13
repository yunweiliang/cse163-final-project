import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

    def setX(self, x = None):
        if x is None:
            self._X = self._data.loc[:, self._data.columns != 'prediction']
            self._X = pd.get_dummies(self._X)
        else:
            self._X = x

    def sety(self, y=None):
        if y is None:
            self._y = self._data['prediction']
        else:
            self._y = y
    
    def decision_tree(self, x=None, y=None):
        self.setX(x=x)
        self.sety(y=y)
        X_train, X_test, y_train, y_test = train_test_split(self._X, self._y, test_size=0.2)
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        result = accuracy_score(y_test, model.predict(X_test))
        return result

    def naive_bayes(self, x=None, y=None):
        self.setX(x=x)
        self.sety(y=y)
        X_train, X_test, y_train, y_test = train_test_split(self._X, self._y, test_size=0.2)
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        score_test = accuracy_score(y_test, y_test_pred)
        return score_test

    def forest(self, x=None, y=None):
        data =  self._data.drop('prediction', axis=1)
        col_name = list(data.columns)
 
        self.setX(x=np.array(self._X))
        self.sety(y=np.array(self._y))

        X_train, X_test, y_train, y_test = train_test_split(self._X, self._y, test_size=0.2)
        rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)
        rf_model.fit(X_train, y_train)

        # save feature importances for Part 2: feature selection
        importances = list(rf_model.feature_importances_)
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(col_name, importances)]
        self._feature_importances = sorted(feature_importances, key=lambda x:x[1], reverse=True)
        #[print('Variable: {} Importance: {}'.format(*pair)) for pair in feature_importances]

        return rf_model.score(X_test, y_test)

    def _run_trials(self, n=10):
        data = {'decision_tree':np.zeros(n),
                'naive_bayes':np.zeros(n),
                'forest':np.zeros(n)}
        for i in range(n):  
            data['decision_tree'][i] = self.decision_tree()
            data['naive_bayes'][i] = self.naive_bayes()
            data['forest'][i] = self.forest()

        return pd.DataFrame(data)

    def calculate_mean_accuracy(self, n=10):
        trials_df = self._run_trials(n=n)
        means = {'decision_tree': sum(trials_df.loc[:,'decision_tree']) / n,
                     'naive_bayes': sum(trials_df.loc[:,'naive_bayes']) / n,
                     'forest': sum(trials_df.loc[:,'forest'])/n}
        return means

    def trials_box_plot(self, n=10):
        sns.set(style='whitegrid')
        sns.boxplot(data=self._run_trials())
        plt.title('Model Performance over ' + str(n) + ' Trials')
        plt.xticks((0, 1, 2), ('Decision Tree', 'Naive Bayes', 'Random Forest'))
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.savefig('box_plot.png')

def main():
    model = ML_Model('cleveland_processed.csv')

    print('Decision Tree Score:', model.decision_tree())
    print('Gaussian Naive Bayes Score:', model.naive_bayes())
    print('Random Forest Score:', model.forest())
    print()
    mean_accuracy = model.calculate_mean_accuracy()
    print('Decision Tree Mean Score:', mean_accuracy['decision_tree'])
    print('Gaussian Naive Bayes Mean Score:', mean_accuracy['naive_bayes'])
    print('Random Forest Mean Score:', mean_accuracy['forest'])


    model.trials_box_plot()
 
if __name__ == '__main__':
    main()