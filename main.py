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
        print(i)
        x = np.array(data.loc[:, i])
        corr[i] = pearsonr(x,y)
    print(corr)

    
def main():
    data = pd.read_csv('cleveland_processed.csv')
    model = ML_Model(data)
    clean_data = model.get_clean_data()
    correlation(clean_data)
    

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