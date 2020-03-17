# Phuong Vu, Yunwei Liang, Trinh Nguyen
#
# analyze_ml_models contains the ML_Model
# class with functions for analysis on three
# Machine Learning models: Decision Tree Classifier,
# Random Forest Classifier, and Gaussian Naive Bayes
# Classifier.

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
from sklearn.tree import export_graphviz

from sklearn.model_selection import KFold, cross_val_score


class ML_Model:
    def __init__(self, df):
        """
        Takes in a DataFrame of heart disease data
        and create a ML_Model object
        """
        self._data = df
        self._data = self._data.dropna().reset_index(drop=True)
        self._data.loc[self._data['prediction'] > 0,
                       'prediction'] = 1  # Make into binary problem
        self._feature_importances = None

    def get_clean_data(self):
        """
        Returns a processed data with prediction converted to binary values.
        """
        return self._data

    def decision_tree(self, x_exempt=None):
        """
        Takes in an optional x_exempt argument and trains a Decision Tree
        Classifier with all attributes except x_exempt as the features
        and prediction as the label. Outputs a decision_tree_model.pdf
        of the model visualization. Returns the accuracy of model.
        """
        X = self._data.loc[:, self._data.columns != 'prediction']
        if x_exempt is not None:
            X = X.loc[:, X.columns != x_exempt]
        X = pd.get_dummies(X)
        y = self._data['prediction']

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2)
        model = DecisionTreeClassifier()
        model = model.fit(X_train, y_train)

        graph = self._plot_tree(model, X=X, y=y)
        graph.render('decision_tree_model')  # save as pdf

        score = accuracy_score(y_test, model.predict(X_test))
        return score

    def forest(self, x_exempt=None):
        """
        Takes in an optional x_exempt argument and trains a Random Forest
        Classifier with all attributes except x_exempt as the features
        and prediction as the label. Outputs a random_forest_model.pdf
        of the model visualization. Returns the accuracy of model.
        """
        # save copy of dataframe format for plotting
        X = self._data.loc[:, self._data.columns != 'prediction']
        if x_exempt is not None:
            X = X.loc[:, X.columns != x_exempt]
        columns = X.columns
        X = pd.get_dummies(X)
        y = self._data['prediction']
        # select info and change to numpy format for random forest modeling
        X_np = np.array(X)
        y_np = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X_np, y_np,
                                                            test_size=0.2)
        model = RandomForestClassifier(n_estimators=1000, random_state=42)
        model.fit(X_train, y_train)

        graph = self._plot_tree(model.estimators_[10],
                                X=X, y=y)  # only select five trees to display
        graph.render('random_forest_model')

        # save feature importances for future analysis
        importances = list(model.feature_importances_)
        feature_importances = [(feature, round(importance, 2)) for feature,
                               importance in zip(list(columns), importances)]
        self._feature_importances = sorted(feature_importances,
                                           key=lambda x: x[1], reverse=True)
        # [print('Variable: {} Importance: {}'.format(*pair)) \
        # for pair in feature_importances]

        score = accuracy_score(y_test, model.predict(X_test))
        return score

    def naive_bayes(self, x_exempt=None):
        """
        Takes in an optional x_exempt argument and trains a Gaussian Naive
        Bayes Classifier with all attributes except x_exempt as the features
        and prediction as the label. Returns the accuracy of model.
        """
        X = self._data.loc[:, self._data.columns != 'prediction']
        if x_exempt is not None:
            X = X.loc[:, X.columns != x_exempt]
        X = pd.get_dummies(X)
        y = self._data['prediction']

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2)
        model = GaussianNB()
        model.fit(X_train, y_train)

        score = accuracy_score(y_test, model.predict(X_test))
        return score

    def calculate_mean_accuracy(self, n=10, x_exempt=None):
        """
        Takes in optional n and optional x_exempt arguments. Runs n trials
        for each of the three models with all attributes except x_exempt
        as the features and prediction as the label. Returns a dictionary
        of model type mapped to their average accuracy scores.
        """
        trials_df = self.run_trials(n=n, x_exempt=x_exempt)
        means = {'decision_tree': sum(trials_df.loc[:, 'decision_tree']) / n,
                 'forest': sum(trials_df.loc[:, 'forest'])/n,
                 'naive_bayes': sum(trials_df.loc[:, 'naive_bayes']) / n}
        return means

    def models_performances_box_plot(self, n=10):
        """
        Takes in an optional n argument. Runs n trials for each of the
        three models with all attributes as the features and prediction
        as the label. Outputs a box_plot.png file of box plots comparing
        the accuracy scores of the three models. Returns a dictionary
        of model type mapped to their average accuracy scores.
        """
        sns.set(style='whitegrid')
        trials_df = self.run_trials()
        sns.boxplot(data=trials_df)
        means = {'decision_tree': sum(trials_df.loc[:, 'decision_tree']) / n,
                 'forest': sum(trials_df.loc[:, 'forest']) / n,
                 'naive_bayes': sum(trials_df.loc[:, 'naive_bayes']) / n}
        plt.title('Model Performance over ' + str(n) + ' Trials')
        plt.xticks((0, 1, 2), ('Decision Tree',
                               'Random Forest',
                               'Naive Bayes'))
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.savefig('box_plot.png')
        return means

    def run_trials(self, n=10, x_exempt=None):
        """
        Takes in optional n and optional x_exempt arguments. Runs n trials
        for each of the three models with all attributes except x_exempt
        as the features and prediction as the label. Returns a DataFrame
        of model as column names and each trial's accuracy recorded in
        respective columns.
        """
        data = {'decision_tree': np.zeros(n),
                'forest': np.zeros(n),
                'naive_bayes': np.zeros(n)}
        for i in range(n):
            data['decision_tree'][i] = self.decision_tree(x_exempt)
            data['naive_bayes'][i] = self.naive_bayes(x_exempt)
            data['forest'][i] = self.forest(x_exempt)

        return pd.DataFrame(data)

    def cross_validation(self, x_exempt=None):
        """
        Takes in optional n and optional x_exempt arguments. Runs n trials
        for each of the three models with all attributes except x_exempt
        as the features and prediction as the label. Returns a DataFrame
        of model as column names and each trial's accuracy recorded in
        respective columns.
        """
        sum = 0
        X = self._data.loc[:, self._data.columns != 'prediction']
        if x_exempt is not None:
            X = X.loc[:, X.columns != x_exempt]
        X = pd.get_dummies(X)
        y = self._data['prediction']

        k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
        model = GaussianNB()
        val = cross_val_score(model, X, y, cv=k_fold, n_jobs=1)
        for i in val:
            sum += i
        return sum/len(val)

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