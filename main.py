# Phuong Vu, Yunwei Liang, Trinh Nguyen
#
# main contains functions for feature analysis and
# runs functions from ML_Model class for model analysis

from analyze_ml_models import ML_Model
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt


def correlation(clean_data):
    corr = {}
    y = np.array(clean_data.loc[:, 'prediction'])
    data = clean_data.drop('prediction', axis=1)
    col_name = list(data.columns)
    for i in col_name:
        x = np.array(data.loc[:, i])
        corr[i] = pearsonr(x, y)
    df = pd.DataFrame(corr, index=['Correlation Coefficient',
                                   '2 Tailed P-Value'])
    return df


def plot_feature_importance(model, xes_exempt):
    df = pd.DataFrame(columns=['Feature', 'Contains Feature',
                               'Decision Tree', 'Random Forest',
                               'Naive Bayes'])
    n = 5  # default five trials

    # comment next 2 lines to run for all features
    # xes_exempt = ['age', 'sex']
    # n = 1

    for x_exempt in xes_exempt:
        with_dict = model.calculate_mean_accuracy(n=n, x_exempt=x_exempt)
        without_dict = model.calculate_mean_accuracy(n=n, x_exempt=x_exempt)
        with_row = {'Feature': x_exempt, 'Contains Feature': True,
                    'Decision Tree': with_dict['decision_tree'],
                    'Random Forest': with_dict['forest'],
                    'Naive Bayes': with_dict['naive_bayes']}
        without_row = {'Feature': x_exempt, 'Contains Feature': False,
                       'Decision Tree': without_dict['decision_tree'],
                       'Random Forest': without_dict['forest'],
                       'Naive Bayes': without_dict['naive_bayes']}
        df = df.append(with_row, ignore_index=True)
        df = df.append(without_row, ignore_index=True)
    df = df.melt(id_vars=['Feature', 'Contains Feature'],
                 value_vars=['Decision Tree', 'Random Forest', 'Naive Bayes'],
                 var_name='model', value_name='Accuracy Score')
    graph = sns.catplot(x='Feature', y='Accuracy Score', col='model', data=df,
                        kind='bar', hue='Contains Feature', col_wrap=3)
    graph = graph.set_xticklabels(rotation=30)
    plt.subplots_adjust(top=0.9)
    graph.fig.suptitle('Performance of Models with vs. without a Feature')
    plt.savefig('features_performances_in_models.png')
    return df


def main():
    data = pd.read_csv('cleveland_processed.csv')
    model = ML_Model(data)
    clean_data = model.get_clean_data()
    clean_data.to_csv('clean_data.csv', index=False)

    print('Decision Tree Score:', model.decision_tree())
    print('Gaussian Naive Bayes Score:', model.naive_bayes())
    print('Random Forest Score:', model.forest())
    print()

    # Following function calls runs multiple trials
    # Comment out if avoiding time-consuming operations
    # e.g. for testing purposes
    mean_accuracy = model.calculate_mean_accuracy()
    print('Decision Tree Mean Score:', mean_accuracy['decision_tree'])
    print('Gaussian Naive Bayes Mean Score:', mean_accuracy['naive_bayes'])
    print('Random Forest Mean Score:', mean_accuracy['forest'])
    print()

    print('Box Plot Means:', model.models_performances_box_plot())
    print()

    feature_correlations = correlation(clean_data)
    print(feature_correlations)
    feature_correlations.to_csv('feature_correlations_to_prediction.csv')
    print()

    print('Cross Validation:', model.cross_validation())
    print()

    feature_importances = plot_feature_importance(model,
        clean_data.columns[clean_data.columns != 'prediction'])
    print(feature_importances)
    feature_importances.to_csv('features_performances_in_models.csv')


if __name__ == '__main__':
    main()