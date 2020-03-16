import pandas as pd 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB

def main():
    cleveland = pd.read_csv('cleveland_processed.csv')
    cleveland = cleveland.dropna().reset_index(drop=True)
    cleveland.loc[cleveland['prediction'] > 0, 'prediction'] = 1
   

    X = cleveland.loc[:, cleveland.columns != 'prediction']
    X = pd.get_dummies(X)
    y = cleveland['prediction']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    score_train = accuracy_score(y_train, y_train_pred)
    score_test = accuracy_score(y_test, y_test_pred)
    #score_train = model.score(X_train, y_train)
    #score_test = model.score(X_test, y_test)

    print('Score Train:', score_train)
    print('Score Test:', score_test)

if __name__ == '__main__':
    main()