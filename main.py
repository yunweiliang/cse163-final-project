from analyze_ml_models import ML_Model
import numpy as np
import pandas as pd
from scipy.stats import pearsonr 
import seaborn as sns
import matplotlib.pyplot as plt

def correlation(clean_data):
    corr = {}
    y = np.array(clean_data.loc[:, 'prediction'])
    data =  clean_data.drop('prediction', axis=1)
    col_name = list(data.columns)
    for i in col_name:
        x = np.array(data.loc[:, i])
        corr[i] = pearsonr(x,y)
    print('Correlations:', corr)

def plot_feature_importance(model, xes_exempt):
    df = pd.DataFrame(columns=['feature', 'contains', 'tree_mean', 'forest_mean', 'bayes_mean'])
    # comment next line to run for all features
    xes_exempt = ['age', 'sex'] 
    for x_exempt in xes_exempt:
        with_dict = model.calculate_mean_accuracy(n=5, x_exempt=x_exempt)
        without_dict = model.calculate_mean_accuracy(n=5, x_exempt=x_exempt)
        with_row = {'feature':x_exempt, 'contains':True, 
                    'tree_mean':with_dict['decision_tree'],
                    'forest_mean':with_dict['forest'],
                    'bayes_mean':with_dict['naive_bayes']}
        without_row = {'feature':x_exempt, 'contains':False, 
                    'tree_mean':without_dict['decision_tree'],
                    'forest_mean':without_dict['forest'],
                    'bayes_mean':without_dict['naive_bayes']}
        df = df.append(with_row, ignore_index=True)
        df = df.append(without_row, ignore_index=True)
    #df = df.melt(id_vars=['feature', 'contains'],
                 #value_vars=['tree_mean', 'forest_mean', 'bayes_mean'],
                 #var_name='model', value_name='value')
    sns.catplot(x='feature', y='tree_mean', data=df,
                kind='bar', hue='contains').set_xticklabels(rotation=45)
    plt.title('Performance of Decision Tree with versus without a Feature')
    plt.xlabel('Features')
    plt.ylabel('Accuracy Score')
    plt.savefig('features_performances_tree_model.png')
    
    sns.catplot(x='feature', y='forest_mean', data=df,
                kind='bar', hue='contains').set_xticklabels(rotation=45)
    plt.title('Performance of Random Forest with versus without a Feature')
    plt.xlabel('Features')
    plt.ylabel('Accuracy Score')
    plt.savefig('features_performances_forest_model.png')

    sns.catplot(x='feature', y='forest_mean', data=df,
                kind='bar', hue='contains').set_xticklabels(rotation=45)
    plt.title('Performance of Naive Bayes with versus without a Feature')
    plt.xlabel('Features')
    plt.ylabel('Accuracy Score')
    plt.savefig('features_performances_bayes_model.png')

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
    #mean_accuracy = model.calculate_mean_accuracy()
    #print('Decision Tree Mean Score:', mean_accuracy['decision_tree'])
    #print('Gaussian Naive Bayes Mean Score:', mean_accuracy['naive_bayes'])
    #print('Random Forest Mean Score:', mean_accuracy['forest'])

    #print(model.models_performances_box_plot())
    #print()

    #correlation(clean_data)
    #print()
    
    # Comment out 
    plot_feature_importance(model, clean_data.columns[clean_data.columns != 'prediction'])

if __name__ == '__main__':
    main()