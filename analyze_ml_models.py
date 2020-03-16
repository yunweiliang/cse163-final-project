import pandas as pd
import numpy as np
import graphviz
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image

class ML_Model:
    def __init__(self, file_path):
        self._data = file_path
        self._data = self._data.dropna().reset_index(drop=True)
        self._data.loc[self._data['prediction'] > 0, 'prediction'] = 1 # Make into binary problem
        self._feature_importances = None

    def get_clean_data(self):
        return self._data
    
    def decision_tree(self, x_exempt=None):
        X = self._data.loc[:, self._data.columns != 'prediction']
        if x_exempt is not None:
            X = self._data.loc[:, self._data.columns != x_exempt]
        X = pd.get_dummies(X)
        y = self._data['prediction']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = DecisionTreeClassifier()
        model = model.fit(X_train, y_train)

        graph = self._plot_tree(model, X=X, y=y)
        graph.render('decision_tree_model') # save as pdf
        
        score = accuracy_score(y_test, model.predict(X_test))
        return score

    def naive_bayes(self, x_exempt=None):
        X = self._data.loc[:, self._data.columns != 'prediction']
        if x_exempt is not None:
            X = self._data.loc[:, self._data.columns != x_exempt]
        X = pd.get_dummies(X)
        y = self._data['prediction']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = GaussianNB()
        model.fit(X_train, y_train)

        score = accuracy_score(y_test, model.predict(X_test))
        return score

    def forest(self, x_exempt=None):
        # save copy of dataframe format for plotting
        X = self._data.loc[:, self._data.columns != 'prediction']
        if x_exempt is not None:
            X = self._data.loc[:, self._data.columns != x_exempt]
        columns = X.columns
        X = pd.get_dummies(X)
        y = self._data['prediction']
        # select info and change to numpy format for random forest modeling
        X_np = np.array(X)
        y_np = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2)
        model = RandomForestClassifier(n_estimators=1000, random_state=42)
        model.fit(X_train, y_train)
        
        graph = self._plot_tree(model.estimators_[5], X=X, y=y) # only select five trees to display
        graph.render('random_forest_model')
        
        # save feature importances for future analysis
        importances = list(model.feature_importances_)
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(list(columns), importances)]
        self._feature_importances = sorted(feature_importances, key=lambda x:x[1], reverse=True)
        #[print('Variable: {} Importance: {}'.format(*pair)) for pair in feature_importances]

        score = accuracy_score(y_test, model.predict(X_test))
        return score

    def calculate_mean_accuracy(self, n=10, x_exempt=None):
        trials_df = self.run_trials(n=n, x_exempt=x_exempt)
        means = {'decision_tree': sum(trials_df.loc[:,'decision_tree']) / n,
                     'naive_bayes': sum(trials_df.loc[:,'naive_bayes']) / n,
                     'forest': sum(trials_df.loc[:,'forest'])/n}
        return means

    def models_performances_box_plot(self, n=10):
        sns.set(style='whitegrid')
        sns.boxplot(data=self.run_trials())
        plt.title('Model Performance over ' + str(n) + ' Trials')
        plt.xticks((0, 1, 2), ('Decision Tree', 'Naive Bayes', 'Random Forest'))
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.savefig('box_plot.png')
     
    def run_trials(self, n=10, x_exempt=None):
        data = {'decision_tree':np.zeros(n),
                'naive_bayes':np.zeros(n),
                'forest':np.zeros(n)}
        for i in range(n):  
            data['decision_tree'][i] = self.decision_tree(x_exempt)
            data['naive_bayes'][i] = self.naive_bayes(x_exempt)
            data['forest'][i] = self.forest(x_exempt)

        return pd.DataFrame(data)
    
    def _plot_tree(self, model, X, y):
        """
        This function takes a model and the X and y values for a dataset
        and plots a visualization of the decision tree

        This function won't work with your cse163 environment.
        """
        dot_data = export_graphviz(model, out_file=None, 
                            feature_names=X.columns,  
                            class_names=str(y.unique()),  
                            filled=True, rounded=True,  
                            special_characters=True) 
        return graphviz.Source(dot_data)