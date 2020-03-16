from analyze_ml_models import ML_Model
import numpy as np
import pandas as pd
from scipy.stats import pearsonr 

def correlation(clean_data):
    corr = {}
    y = np.array(clean_data.loc[:, 'prediction'])
    data =  clean_data.drop('prediction', axis=1)
    col_name = list(data.columns)
    for i in col_name:
        x = np.array(data.loc[:, i])
        corr[i] = pearsonr(x,y)
    print('Correlations:', corr)

def plot_feature_importance(model, x_exempt):
    with_dict = model.calculate_mean_accuracy(x_exempt=x_exempt)
    without_dict = model.calculate_mean_accuracy(x_exempt=x_exempt)
    combined = [with_dict, without_dict]
    df = pd.DataFrame(combined)
    print(df)
    df = df.transpose()
    print(df)
    
def main():
    data = pd.read_csv('cleveland_processed.csv')
    model = ML_Model(data)
    clean_data = model.get_clean_data()
    clean_data.to_csv('clean_data.csv', index=False)
    
    print('Decision Tree Score:', model.decision_tree())
    print('Gaussian Naive Bayes Score:', model.naive_bayes())
    print('Random Forest Score:', model.forest())
    print()
    mean_accuracy = model.calculate_mean_accuracy()
    print('Decision Tree Mean Score:', mean_accuracy['decision_tree'])
    print('Gaussian Naive Bayes Mean Score:', mean_accuracy['naive_bayes'])
    print('Random Forest Mean Score:', mean_accuracy['forest'])

    print(model.models_performances_box_plot())

    correlation(clean_data)

    plot_feature_importance(model, 'age')
if __name__ == '__main__':
    main()