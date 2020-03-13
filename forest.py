import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def forest(df):
    df = df.dropna().reset_index(drop=True)
    df.loc[df['prediction'] > 0, 'prediction'] = 1
    #for i in range(len(df['prediction'])):
     #   if df.loc[i, 'prediction'] > 0:
    #        df.loc[i, 'prediction'] = 1
    #print(df.loc[:, 'prediction'])
    # data  prep
    np_label  =  np.array(df['prediction'])
    data =  df.drop('prediction', axis=1)
    col_name = list(data.columns)
    np_data = np.array(data)
    # split data
    X_train, X_test, y_train, y_test = train_test_split(np_data, np_label, test_size=0.2)
    # Train model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    print(rf_model.score(X_train, y_train))
    print(rf_model.score(X_test, y_test))


    importances = list(rf_model.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(col_name, importances)]
    feature_importances = sorted(feature_importances, key=lambda x:x[1], reverse=True)
    #[print('Variable: {} Importance: {}'.format(*pair)) for pair in feature_importances]





def main():
    df = pd.read_csv('filtered.csv')
    forest(df)


if __name__ ==  '__main__':
    main()