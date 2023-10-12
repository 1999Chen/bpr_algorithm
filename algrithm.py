# import export as export
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json

from collections import defaultdict
import sys
# user_preference_data = pd.read_csv("C:/Users/asus/Desktop/data.csv")
final_data = pd.read_csv("C:/Users/asus/Desktop/data.csv")
predict_data=pd.read_csv("C:/Users/asus/Desktop/items.csv")
pd.set_option('display.max_columns', None)
# import
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("game num")
    plt.ylabel("score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def getPrediction(age,gender,region):

    # # penalty = 'l1', C = 3,

    X = final_data
    print("getting x")
    y = X['is_prefered']
    print(final_data)
    X = X.drop(['is_prefered'], axis='columns')


    X = pd.get_dummies(X, columns=['user_region', 'item_category', 'item_region'])

    print(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=69,
                                                        test_size=0.25,
                                                        shuffle=True)

    # penalty = 'l1', C = 3,

    logreg = LogisticRegression(solver='liblinear', multi_class='auto')
    logreg.fit(X_train, y_train)
    score = logreg.score(X_train, y_train)
    score2 = logreg.score(X_test, y_test)
    # return user_preference_data

    y_pred = logreg.predict(X_test)
    score = logreg.score(X_train, y_train)
    score2 = logreg.score(X_test, y_test)


    p1 = final_data.drop(index=final_data.index)

    predict_data['user_age']=age
    predict_data['user_gender'] = gender
    predict_data['user_region'] = region

    # predict_data
    X1 = pd.get_dummies(predict_data, columns=['user_region', 'item_category', 'item_region'])
    X1=X1.drop(['item_id' , 'is_prefered'], axis='columns')

    X1.insert(loc=2, column='user_region_Denmark', value=0)
    X1.insert(loc=3, column='user_region_Germany', value=0)
    X1.insert(loc=4, column='user_region_Japan', value=0)
    X1.insert(loc=5, column='user_region_Korea', value=0)

    list = logreg.predict(X1).tolist()
    #
    # count= 0
    # for x in list:
    #     if x == 1
    #
    #     count = count+1
    itemList = [i for i,x in enumerate(list) if x==1]
    print(itemList)
    result = []
    for x in itemList:
        result.append(predict_data['item_id'][x])
        print(predict_data['item_id'][x])
    print(result)
    result = json.dumps(result, default=str)

    return result