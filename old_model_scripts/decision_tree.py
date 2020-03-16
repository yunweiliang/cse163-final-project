import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def decision_tree(data):
    data = data.dropna().reset_index(drop=True)
    data.loc[data['prediction'] > 0, 'prediction'] = 1
    X = data.loc[:, data.columns != 'prediction']
    X = pd.get_dummies(X)
    y = data['prediction']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    result = accuracy_score(y_test, model.predict(X_test))
    return result


def main():
    data = pd.read_csv('cleveland_processed.csv')
    print(decision_tree(data))

if __name__ ==  '__main__':
    main()