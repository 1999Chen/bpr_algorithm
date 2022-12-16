import export as export
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

user_preference_data = pd.read_csv("C:/Users/lsy19/Desktop/data.csv")
final_data = pd.read_csv("C:/Users/lsy19/Desktop/data_temp1.csv")


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


def getPrediction():
    print("user_preference_data")
    # print(user_preference_data)
    # df =  user_preference_data
    # df1 = pd.get_dummies(df['user_region'], dummy_na = True)
    # # df2 = pd.get_dummies(df['user_region'], dummy_na=True)
    # df2 = pd.get_dummies(df['item_category'], dummy_na=True)
    # df3 = pd.get_dummies(df['item_region'], dummy_na=True)
    # result1 = pd.concat([df, df1], axis=1)
    # result2 = pd.concat([result1, df2], axis=1)
    # result3 = pd.concat([result2, df3], axis=1)
    # print(result3)
    # path = "C:/Users/lsy19/Desktop/data_temp1.csv"

    # result3.to_csv(path, index_label="index_label")
    X = final_data
    y = X['is_prefered']
    X = X.drop(["user_region", "item_region", 'item_category', 'is_prefered'], axis='columns')

    print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=69,
                                                        test_size=0.25,
                                                        shuffle=True)
    # logreg = LogisticRegression(penalty='l1', C=0.8, solver='liblinear', multi_class='auto')
    # logreg = LogisticRegression()
    # logreg.fit(X_train, y_train)
    # score = logreg.score(X_train, y_train)
    # score2 = logreg.score(X_test, y_test)
    # # return user_preference_data
    # print("score 1 is ", score)
    # print("score 2 is", score2)
    #
    # logreg = LogisticRegression(penalty='l1', C=0.8, solver='liblinear', multi_class='auto')
    # logreg.fit(X_train, y_train)
    # score = logreg.score(X_train, y_train)
    # score2 = logreg.score(X_test, y_test)
    # # return user_preference_data
    # print("score 1 is ", score)
    # print("score 2 is", score2)

    logreg = LogisticRegression(penalty='l2', C=2, solver='liblinear', multi_class='auto')
    logreg.fit(X_train, y_train)
    score = logreg.score(X_train, y_train)
    score2 = logreg.score(X_test, y_test)
    # return user_preference_data

    y_pred = logreg.predict(X_test)
    # # return user_preference_data
    # print("score 1 is ", score)
    # print("score 2 is", score2)
    # print("ACC", accuracy_score(y_test, y_pred))
    # print("REC", recall_score(y_test, y_pred, average="micro"))
    # print("F-score", f1_score(y_test, y_pred, average="micro"))
    #
    # logreg = LogisticRegression(penalty='l1', C=1.1, solver='liblinear', multi_class='auto')
    # logreg.fit(X_train, y_train)
    # score = logreg.score(X_train, y_train)
    # score2 = logreg.score(X_test, y_test)
    # y_pred = logreg.predict(X_test)
    # # return user_preference_data
    # print("score 1 is ", score)
    # print("score 2 is", score2)
    # print("ACC", accuracy_score(y_test, y_pred))
    # print("REC", recall_score(y_test, y_pred, average="micro"))
    # print("F-score", f1_score(y_test, y_pred, average="micro"))
    # estimator = PCA(n_components=10)     # dimensions from 19 to 10
    #
    # pca_X_train = estimator.fit_transform(X_train)
    #
    #
    # pca_X_test = estimator.transform(X_test)
    #
    # logreg = LogisticRegression(penalty='l1', C=0.8, solver='liblinear', multi_class='auto')
    # logreg.fit(pca_X_train, y_train)
    # score = logreg.score(pca_X_train, y_train)
    # score2 = logreg.score(pca_X_test, y_test)
    # y_pred = logreg.predict(pca_X_test)
    # return user_preference_data

    # print("score 1 is ", score)
    # print("score 2 is", score2)
    print("ACC", accuracy_score(y_test, y_pred))
    print("REC", recall_score(y_test, y_pred, average="micro"))
    print("F-score", f1_score(y_test, y_pred, average="micro"))
    # cv = ShuffleSplit(n_splits=1000, test_size=0.2, random_state=0)
    # plot_learning_curve(logreg, "logreg", X_train, y_train, ylim=None, cv=cv, n_jobs=1)

    return logreg.predict(X_test).tolist()