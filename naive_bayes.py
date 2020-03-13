import pandas as pd 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    cleveland = pd.read_csv('copy_filtered - cleveland_processed.csv')
    cleveland = cleveland.dropna()
    X = cleveland.loc[:, cleveland.columns != 'prediction']
    X = pd.get_dummies(X)
    y = cleveland['prediction']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)

    for (x, y) in (zip(y_train_pred, y_train)):
        print('(' + str(x) + ', ' + str(y) + ')')

    print(accuracy_score(y_train, y_train_pred))
if __name__ == '__main__':
    main()