import matplotlib.pyplot as plt
import pandas as pd
import plotly.figure_factory as ff
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

model_list = [
    "LDA",
    "RandomForest",
    "DecisionTree",
    "XGBoostTree",
    "LogisticRegression",
]
Accuracy1 = []
Accuracy2 = []
cross = []


def ROC_AUC(y_test, y_predict, y_prob, model):
    score = roc_auc_score(y_test, y_predict)
    f, t, thresholds = roc_curve(y_test, y_prob[:, 1])
    plt.figure()
    plt.plot(f, t, label=f"{model} (area = %0.2f)" % score)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC")
    plt.legend(loc="lower right")
    plt.savefig(f"plot/roc/{model}")


def plot_confusion_matrix(y_test, y_predict, model):
    cm = confusion_matrix(y_test, y_predict)
    x = ["0", "1"]
    y = ["0", "1"]
    fig = ff.create_annotated_heatmap(cm, x=x, y=y, colorscale="Viridis")

    # add title
    fig.update_layout(xaxis_title="Predicted value", yaxis_title="Real value")

    fig.update_layout(
        width=600,
        height=600,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        template="none",
    )
    filename = f"plot/confusion_matrix/{model}.html"
    fig.write_html(
        file=filename,
        include_plotlyjs="cdn",
    )


def modelling(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=1234
    )

    # LDA
    pipe = Pipeline([("Normalize", Normalizer()), ("LDA", LDA(n_components=1))])
    pipe.fit(X_train, y_train)
    y_predict = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)
    confusion_mat = confusion_matrix(y_test, y_predict)
    accuracy_LDA = accuracy_score(y_test, y_predict)
    Accuracy1.append(accuracy_LDA)
    print("confusion matrix of LDA:", confusion_mat)
    print("accuracy of LDA", accuracy_LDA)
    print("Classification Report of LDA\n", classification_report(y_test, y_predict))

    plot_confusion_matrix(y_test, y_predict, "LDAClsssifier")
    ROC_AUC(y_test, y_predict, y_prob, "LDAClsssifier")

    # Random Forest Classifier
    random_classifier = Pipeline(
        [
            ("Normalize", Normalizer()),
            (
                "RFC",
                RandomForestClassifier(
                    n_estimators=60,
                    max_depth=4,
                    criterion="entropy",
                    random_state=499,
                ),
            ),
        ]
    )
    random_classifier.fit(X_train, y_train)
    y_predict = random_classifier.predict(X_test)
    y_prob = random_classifier.predict_proba(X_test)
    accuracy_RFC = accuracy_score(y_test, y_predict)

    Accuracy1.append(accuracy_RFC)
    print("Accuracy of RFC", accuracy_RFC)
    print(
        "Classification Report of Random Forest Classifier\n",
        classification_report(y_test, y_predict),
    )
    plot_confusion_matrix(y_test, y_predict, "RandomForestClassifier")
    ROC_AUC(y_test, y_predict, y_prob, "RandomForestClassifier")

    # Decision Tree Classifier
    DecisionTree_classifier = Pipeline(
        [
            ("Normalize", Normalizer()),
            (
                "DTC",
                DecisionTreeClassifier(
                    max_features=None,
                    criterion="entropy",
                    max_depth=5,
                    random_state=0,
                ),
            ),
        ]
    )
    DecisionTree_classifier.fit(X_train, y_train)
    y_predict = DecisionTree_classifier.predict(X_test)
    y_prob = DecisionTree_classifier.predict_proba(X_test)
    accuracy_DTC = accuracy_score(y_test, y_predict)

    Accuracy1.append(accuracy_DTC)
    print("Accuracy of Decision Tree Classifier", accuracy_DTC)
    print(
        "Classification Report of Decision Tree Classifier\n",
        classification_report(y_test, y_predict),
    )
    plot_confusion_matrix(y_test, y_predict, "DecisionTressClassifier")
    ROC_AUC(y_test, y_predict, y_prob, "DecisionTressClassifier")

    # XGBoost
    XGBoostTree_classifier = Pipeline(
        [
            ("Normalize", Normalizer()),
            (
                "XGB",
                XGBClassifier(
                    max_depth=3,
                    learning_rate=0.05,
                    n_estimatores=300,
                    random_state=0,
                    use_label_encoder=False,
                ),
            ),
        ]
    )
    XGBoostTree_classifier.fit(X_train, y_train)
    y_predict = XGBoostTree_classifier.predict(X_test)
    y_prob = XGBoostTree_classifier.predict_proba(X_test)
    accuracy_XGB = accuracy_score(y_test, y_predict)

    Accuracy1.append(accuracy_XGB)
    print("Accuracy of Decision Tree Classifier", accuracy_XGB)
    print(
        "Classification Report of XGBoost Tree Classifier\n",
        classification_report(y_test, y_predict),
    )
    plot_confusion_matrix(y_test, y_predict, "XGBoostClassifier")
    ROC_AUC(y_test, y_predict, y_prob, "XGBoostClassifier")

    # Logistic Regression

    logistic = LogisticRegression(solver="lbfgs", max_iter=1000)
    logistic.fit(X_train, y_train)
    y_predict = logistic.predict(X_test)
    y_prob = logistic.predict_proba(X_test)
    accuracy_LR = accuracy_score(y_test, y_predict)
    Accuracy1.append(accuracy_LR)
    print("Accuracy of Logistic Regression", accuracy_LR)
    print(
        "Classification Report of Logistic Regression\n",
        classification_report(y_test, y_predict),
    )
    plot_confusion_matrix(y_test, y_predict, "LogisticRegressionClassifier")
    ROC_AUC(y_test, y_predict, y_prob, "LogisticRegressionClassifier")

    model = []
    i = 0
    for m in model_list:
        model.append(
            dict(
                Model=m,
                Accuracy=Accuracy1[i],
            )
        )
        i += 1
    df1 = pd.DataFrame(model)

    html = df1.to_html(escape=True, render_links=True)
    text_file = open("plot/Model_Accuracy.html", "w")
    text_file.write(html)
    text_file.close()


def modelling1(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=1234
    )

    # LDA
    pipe = Pipeline([("Normalize", Normalizer()), ("LDA", LDA(n_components=1))])
    pipe.fit(X_train, y_train)
    y_predict = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)
    confusion_mat = confusion_matrix(y_test, y_predict)
    accuracy_LDA = accuracy_score(y_test, y_predict)
    Accuracy2.append(accuracy_LDA)
    print("confusion matrix of LDA:", confusion_mat)
    print("accuracy of LDA", accuracy_LDA)
    print("Classification Report of LDA\n", classification_report(y_test, y_predict))

    # Random Forest Classifier
    random_classifier = Pipeline(
        [
            ("Normalize", Normalizer()),
            (
                "RFC",
                RandomForestClassifier(
                    n_estimators=60,
                    max_depth=4,
                    criterion="entropy",
                    random_state=499,
                ),
            ),
        ]
    )
    random_classifier.fit(X_train, y_train)
    y_predict = random_classifier.predict(X_test)
    y_prob = random_classifier.predict_proba(X_test)
    accuracy_RFC = accuracy_score(y_test, y_predict)

    Accuracy2.append(accuracy_RFC)
    print("Accuracy of RFC", accuracy_RFC)
    print(
        "Classification Report of Random Forest Classifier\n",
        classification_report(y_test, y_predict),
    )

    # Decision Tree Classifier
    DecisionTree_classifier = Pipeline(
        [
            ("Normalize", Normalizer()),
            (
                "DTC",
                DecisionTreeClassifier(
                    max_features=None,
                    criterion="entropy",
                    max_depth=5,
                    random_state=0,
                ),
            ),
        ]
    )
    DecisionTree_classifier.fit(X_train, y_train)
    y_predict = DecisionTree_classifier.predict(X_test)
    y_prob = DecisionTree_classifier.predict_proba(X_test)
    accuracy_DTC = accuracy_score(y_test, y_predict)

    Accuracy2.append(accuracy_DTC)
    print("Accuracy of Decision Tree Classifier", accuracy_DTC)
    print(
        "Classification Report of Decision Tree Classifier\n",
        classification_report(y_test, y_predict),
    )

    # XGBoost
    XGBoostTree_classifier = Pipeline(
        [
            ("Normalize", Normalizer()),
            (
                "XGB",
                XGBClassifier(
                    max_depth=3,
                    learning_rate=0.05,
                    n_estimatores=300,
                    random_state=0,
                    use_label_encoder=False,
                ),
            ),
        ]
    )
    XGBoostTree_classifier.fit(X_train, y_train)
    y_predict = XGBoostTree_classifier.predict(X_test)
    y_prob = XGBoostTree_classifier.predict_proba(X_test)
    accuracy_XGB = accuracy_score(y_test, y_predict)

    Accuracy2.append(accuracy_XGB)
    print("Accuracy of Decision Tree Classifier", accuracy_XGB)
    print(
        "Classification Report of XGBoost Tree Classifier\n",
        classification_report(y_test, y_predict),
    )

    # Logistic Regression

    logistic = LogisticRegression(solver="lbfgs", max_iter=1000)
    logistic.fit(X_train, y_train)
    y_predict = logistic.predict(X_test)
    y_prob = logistic.predict_proba(X_test)
    accuracy_LR = accuracy_score(y_test, y_predict)
    Accuracy2.append(accuracy_LR)
    print("Accuracy of Logistic Regression", accuracy_LR)
    print(
        "Classification Report of Logistic Regression\n",
        classification_report(y_test, y_predict),
    )

    model = []
    i = 0
    for m in model_list:
        model.append(
            dict(
                Model=m,
                Accuracy=Accuracy2[i],
            )
        )
        i += 1
    df1 = pd.DataFrame(model)

    html = df1.to_html(escape=True, render_links=True)
    text_file = open("plot/Model_Accuracy2.html", "w")
    text_file.write(html)
    text_file.close()