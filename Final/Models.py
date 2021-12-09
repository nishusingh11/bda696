import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import cross_val_score


model_list=["LDA", "RandomForest","LogisticRegression","SupportVector"]
Accuracy=[]
cross=[]


def modelling(X,y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=1234
    )

    pipe = Pipeline([("Normalize", Normalizer()), ("LDA", LDA(n_components=1))])
    pipe.fit(X_train, y_train)
    y_predict = pipe.predict(X_test)
    confusion_mat = confusion_matrix(y_test, y_predict)
    accuracy_LDA = accuracy_score(y_test, y_predict)
    Accuracy.append(accuracy_LDA)
    print("confusion matrix of LDA:", confusion_mat)
    print("accuracy of LDA", accuracy_LDA)
    print("Classification Report of LDA\n", classification_report(y_test, y_predict))

    pipe2 = Pipeline([("Normalize", Normalizer()), ("LDA", LDA(n_components=1))])
    score= cross_val_score(pipe2, X,y,cv=4,scoring="accuracy")
    avg_score=np.average(score)
    cross.append(avg_score)

    # Random Forest Classifier
    random_classifier = Pipeline(
        [
            ("Normalize", Normalizer()),
            (
                "RFC",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    criterion="entropy",
                    random_state=499,
                ),
            ),
        ]
    )
    random_classifier.fit(X_train, y_train)
    y_predict = random_classifier.predict(X_test)
    accuracy_RFC = accuracy_score(y_test, y_predict)
    Accuracy.append(accuracy_RFC)
    print("Accuracy of RFC", accuracy_RFC)
    print(
        "Classification Report of Random Forest Classifier\n",
        classification_report(y_test, y_predict),
    )
    pipe3 = Pipeline(
        [
            ("Normalize", Normalizer()),
            (
                "RFC",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    criterion="entropy",
                    random_state=499,
                ),
            ),
        ]
    )
    score = cross_val_score(pipe3, X, y, cv=4, scoring="accuracy")
    avg_score = np.average(score)
    cross.append(avg_score)
    #Logistic Regression

    logistic = LogisticRegression(solver="lbfgs", max_iter=1000)
    logistic.fit(X_train, y_train)
    y_predict = logistic.predict(X_test)
    accuracy_LR = accuracy_score(y_test, y_predict)
    Accuracy.append(accuracy_LR)
    print("Accuracy of Logistic Regression", accuracy_LR)
    print(
        "Classification Report of Logistic Regression\n",
        classification_report(y_test, y_predict),
    )
    pipe4= LogisticRegression(solver="lbfgs", max_iter=1000)
    score = cross_val_score(pipe4, X, y, cv=4, scoring="accuracy")
    avg_score = np.average(score)
    cross.append(avg_score)
    # support vector machine
    support_vector = svm.SVC()
    support_vector.fit(X_train, y_train)
    y_predict = support_vector.predict(X_test)
    accuracy_SV = accuracy_score(y_test, y_predict)
    Accuracy.append(accuracy_SV)
    print("Accuracy of SV", accuracy_SV)
    print(
        "Classification Report of Support Vector Classifier\n",
        classification_report(y_test, y_predict),
    )
    pipe5=svm.SVC()
    score = cross_val_score(pipe5, X, y, cv=4, scoring="accuracy")
    avg_score = np.average(score)
    cross.append(avg_score)
    #
    # NOTE: The accuracy of Support Vector Classifier model is slightly better
    # than Random forest, Logistic Regression and LDA.
    model = []
    i=0
    for m in model_list:
        model.append(
            dict(
                Name=m,
                Accuracy=Accuracy[i],
                #Cross_score=cross[i]
            )
        )
        i+=1
    df1 = pd.DataFrame(model)
    # print(df)

    html = df1.to_html(escape=True, render_links=True)
    text_file = open("Assignment2.html", "w")
    text_file.write(html)
    text_file.close()