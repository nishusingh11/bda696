# Reused the code of Assignment 2 for SQL, Assignment 3 for spark session builder,
# Assignment 4 and 5 for plotting, correlation and MSD

import os
import correlation_score
import link
import msd
import numpy
import pandas as pd
import statsmodels.api
import table
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objs as go
from pyspark import SparkConf, SparkContext, StorageLevel
from pyspark.sql import SparkSession
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

correlation_table = {}
correlation_matrices = []
correlation_plot = []


# Function checking for boolean or continuous Response
def check_for_response(col):
    if len(set(col)) == 2:
        return "boolean"
    else:
        return "continuous"


# Function checking for boolean or continuous Predictor
def check_for_predictor(column):
    if column.dtypes in ["object"] or (column.nunique() / column.count() < 0.05):
        return "categorical"
    else:
        return "continuous"


# Function for plotting Heatmap plot
def bool_response_cat_predictor(feature, column, Y, response, df):
    df = df[[response, column]]
    fig = px.density_heatmap(df, x=column, y=response)
    fig.update_xaxes(title=column)
    fig.update_yaxes(title=response)

    # fig.show()
    file_name = f"plot/hw_4_plot/cr_cat_{column}.html"
    file_name_cat_con = f"plot/hw_4_plot/cr_cat_con_{column}.html"
    fig.write_html(
        file=file_name,
        include_plotlyjs="cdn",
    )
    fig.write_html(
        file=file_name_cat_con,
        include_plotlyjs="cdn",
    )


# Function for plotting distribution plot
def bool_response_con_predictor(dataset_df, response, column, Y):
    dataset_df = dataset_df.astype(float)
    # df['a'] = df['a'].astype(float, errors='raise')
    hist_data = [dataset_df[Y == 0][column], dataset_df[Y == 1][column]]
    group_labels = ["Response=0", "Response=1"]

    # group_labels = ["0", "1"]
    # label_0 = dataset_df[Y == 0][column]
    # label_1 = dataset_df[Y == 1][column]
    # hist_data = [label_0, label_1]
    print(hist_data)
    colors = ["slategray", "magenta"]
    fig = ff.create_distplot(hist_data, group_labels, bin_size=10, colors=colors)
    fig.update_layout(
        title="Continuous Predictor by Boolean Response",
    )
    # fig.show()
    file_name = f"plot/hw_4_plot/cr_con_{column}.html"
    file_name_cat_con = f"plot/hw_4_plot/cr_cat_con_{column}.html"
    fig.write_html(
        file=file_name,
        include_plotlyjs="cdn",
    )
    fig.write_html(
        file=file_name_cat_con,
        include_plotlyjs="cdn",
    )


# Function for plotting Violin plot
def con_response_cat_predictor(dataset_df, column, Y, response):
    group_labels = ["0", "1"]
    fig = go.Figure()
    label_0 = dataset_df[Y == 0][column]
    label_1 = dataset_df[Y == 1][column]
    hist_data = [label_0, label_1]
    for curr_hist, curr_group in zip(hist_data, group_labels):
        fig.add_trace(
            go.Violin(
                x=numpy.repeat(curr_group, len(dataset_df)),
                y=curr_hist,
                name=curr_group,
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig.update_layout(
        title="Continuous Response by Categorical Predictor",
        xaxis_title=response,
        yaxis_title=column,
    )
    # fig.show()
    file_name = f"plot/hw_4_plot/cr_cat_{column}.html"
    file_name_cat_con = f"plot/hw_4_plot/cr_cat_con_{column}.html"
    fig.write_html(
        file=file_name,
        include_plotlyjs="cdn",
    )
    fig.write_html(
        file=file_name_cat_con,
        include_plotlyjs="cdn",
    )


# Function for plotting scatter plot
def con_response_con_predictor(feature, column, y, response):
    df = pd.DataFrame({column: feature, response: y})
    fig = px.scatter(df, x=column, y=response, trendline="ols")
    fig.update_layout(title="Continuous Response by Continuous Predictor")
    # fig.show()
    column = column.replace("/", "")
    file_name = f"plot/hw_4_plot/cr_con_{column}.html"
    file_name_cat_con = f"plot/hw_4_plot/cr_cat_con_{column}.html"
    fig.write_html(
        file=file_name,
        include_plotlyjs="cdn",
    )
    fig.write_html(
        file=file_name_cat_con,
        include_plotlyjs="cdn",
    )


def linear_regression(y, pred, column):
    linear_regression_model = statsmodels.api.OLS(y, pred)
    linear_regression_model_fitted = linear_regression_model.fit()
    print(f"Feature_Name: {column}")
    print(linear_regression_model_fitted.summary())

    # Get statistics
    tval = round(linear_regression_model_fitted.tvalues[1], 6)
    pval = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

    return tval, pval


def logistic_regression(y, pred, column):
    logistic_regression_model = statsmodels.api.Logit(
        y.astype(float), pred.astype(float)
    )
    logistic_regression_model_fitted = logistic_regression_model.fit()
    print(f"Feature_Name:{column}")
    print(logistic_regression_model_fitted.summary())

    # Get the statistics
    tval = round(logistic_regression_model_fitted.tvalues[1], 6)
    pval = "{:.6e}".format(logistic_regression_model_fitted.pvalues[1])

    return tval, pval


def plot1(df, title):
    if df.empty:
        pass
    else:
        fig = ff.create_annotated_heatmap(
            z=df.to_numpy(),
            x=df.columns.tolist(),
            y=df.index.tolist(),
            zmax=1,
            zmin=-1,
            showscale=True,
            hoverongaps=True,
        )

        fig.update_layout(title_text=f"<i><b>{title}</b></i>")
        fig.show()
        return fig


def main():
    if not os.path.exists("plot/brute_force"):
        os.makedirs("plot/brute_force")
    if not os.path.exists("plot/correlation_plot"):
        os.makedirs("plot/correlation_plot")
    if not os.path.exists("plot/hw_4_plot"):
        os.makedirs("plot/hw_4_plot")
    path = os.path.abspath("plot/hw_4_plot")

    spark = (
        SparkSession.builder.master("local[*]")
        .config("spark.sql.debug.maxToStringFields", 128)
        .getOrCreate()
    )

    # Dataframe for final_baseball_features
    dataframe = (
        spark.read.format("jdbc")
        .options(
            url="jdbc:mysql://localhost:3306/baseball",
            driver="org.mariadb.jdbc.Driver",
            dbtable="feature_ratio",
            user="root",
            password="abc123",  # pragma: allowlist secret
        )
        .load()
    )

    dataframe.createOrReplaceTempView("feature_ratio")
    dataframe.persist(StorageLevel.MEMORY_AND_DISK)
    query = spark.sql(
        """ SELECT * FROM feature_ratio
        """
    )
    query.show()
    df = dataframe.toPandas()
    df = df.drop(["home_team_id", "away_team_id", "game_id"], axis=1)
    for column in df:
        df[column].fillna(0, inplace=True)
    df = df.round(3)
    print(df.dtypes)
    df.dropna(axis=1)

    response = "home_team_wins"
    X = df.iloc[:, :-1]
    y = df[response]
    print(X)
    print(y)
    predictor = X.columns
    print(predictor)
    response_type = check_for_response(y)

    if response_type == "boolean":
        df[response] = df[response].astype("category").cat.codes
    y = df[response]
    print(y)

    # Identify Predictor type
    predictor_type = [check_for_predictor(df[i]) for i in predictor]
    print(response_type)
    print(predictor_type)

    # split dataset between categorical and continuous
    df_cat_predictor = pd.DataFrame()
    df_con_predictor = pd.DataFrame()
    cat_names = []
    con_names = []

    for index, column in enumerate(predictor):
        if predictor_type[index] == "continuous":
            df_con_predictor[column] = X[column]
            con_names.append(column)
        else:
            df_cat_predictor[column] = X[column]
            cat_names.append(column)
    print("categorical predictors are:\n", cat_names)
    print("continuous predictors are:\n", con_names)
    print(df_con_predictor)

    for index, column in enumerate(X):
        feature = X[column]
        if predictor_type[index] == "categorical" and response_type == "boolean":
            bool_response_cat_predictor(predictor, column, y, response, df)
        elif predictor_type[index] == "continuous" and response_type == "boolean":
            bool_response_con_predictor(df, response, column, y)
        elif predictor_type[index] == "categorical" and response_type == "continuous":
            con_response_cat_predictor(df, column, y, response)
        elif predictor_type[index] == "continuous" and response_type == "continuous":
            con_response_con_predictor(feature, column, y, response)

    for index, column in enumerate(X):
        feature = X[column]
        pred = statsmodels.api.add_constant(feature)
        if response_type == "continuous" and predictor_type[index] == "continuous":
            t_val, p_val = linear_regression(y, pred, column)
            fig = px.scatter(x=feature, y=y, trendline="ols")
            fig.update_layout(
                title=f"Variable: {column}: (t-value={t_val}) (p-value={p_val})",
                xaxis_title=f"Variable:{column}",
                yaxis_title=response,
            )
            fig.show()
        elif response_type == "boolean" and predictor_type[index] == "continuous":
            t_val, p_val = logistic_regression(y, pred, column)
            fig = px.scatter(x=feature, y=y, trendline="ols")
            fig.update_layout(
                title=f"Variable: {column}: (t-value={t_val}) (p-value={p_val})",
                xaxis_title=f"Variable: {column}",
                yaxis_title=response,
            )
            fig.show()

    # -----------correlation score, matrix and msd table for con-con----------------

    con_con_matrix = df_con_predictor.corr(method="pearson")

    print("\ncon con score matrix\n", con_con_matrix)
    con_con_matrix = con_con_matrix.to_dict()
    con_con_matrix_df = pd.DataFrame(con_con_matrix)

    # Heatmap plot for con con matrix
    con_mat = plot1(
        con_con_matrix_df, "Correlation between Continuous and Continuous predictor"
    )

    correlation_matrices.append(con_con_matrix)

    con_con_table = table.correlation_table(con_con_matrix, response)
    print("\ncon-con correlation table\n", con_con_table)

    # saving con-con matrix in html file
    location = f"plot/correlation_plot/con_con_heatmap.html"
    con_mat.write_html(file=location, default_width="40%", include_plotlyjs="cdn")
    correlation_plot.append(location)

    # con con msd table
    X = X.astype(float)
    con_con_msd_df, con2_diff_plot = msd.con_con_diff(
        X, y, con_names, response, response_type
    )
    print("\ncon con msd table\n", con_con_msd_df)

    # -----------correlation score, matrix and msd table for cat-con----------------
    cat_con_matrix = {}
    if len(con_names) and len(cat_names):
        for i in con_names:
            cat_con_matrix[i] = {}
            for j in cat_names:
                cat_con_matrix[i][j] = round(
                    correlation_score.cat_cont_correlation_ratio(
                        X[j].to_numpy(), X[i].to_numpy()
                    ),
                    5,
                )

    cat_con_matrix_d = pd.DataFrame(cat_con_matrix)

    print("\ncat_con_matrix\n", cat_con_matrix)
    correlation_matrices.append(cat_con_matrix)

    cat_con_table = table.correlation_table(cat_con_matrix, response)
    print("\ncat-con correlation table\n", cat_con_table)

    # Heatmap plot for cat-con matrix
    cat_con_plot = plot1(
        cat_con_matrix_d, "Correlation between Categorical and Continuous predictor"
    )
    # cat_con_mat.show()

    # saving cat-con matrix in html file
    # if type(cat_con_matrix_d) == type(None):
    location = f"plot/correlation_plot/cat_con_heatmap.html"
    cat_con_plot.write_html(file=location, include_plotlyjs="cdn")
    correlation_plot.append(location)

    # cat con msd table
    cat_con_msd_df = msd.cat_con_diff(
        X, y, con_names, cat_names, response, response_type
    )
    print("result cat con msd\n", cat_con_msd_df)

    # -----------correlation score, matrix and msd table for cat-cat----------------

    cat_cat_matrix = {}
    for i in cat_names:
        cat_cat_matrix[i] = {}
        for j in cat_names:
            cat_cat_matrix[i][j] = round(
                correlation_score.cat_correlation(df[i], df[j]), 5
            )

    # print(cat_cat_matrix)
    correlation_matrices.append(cat_cat_matrix)

    cat_cat_table = table.correlation_table(cat_cat_matrix, response)
    print("\ncat-cat correlation table\n", cat_cat_table)

    cat_cat_matrix_df = pd.DataFrame(cat_cat_matrix)

    # Heatmap plot for cat-cat matrix

    cat_cat_mat = plot1(
        cat_cat_matrix_df, "Correlation between Categorical and Categorical predictor"
    )

    # saving cat-cat matrix in html file
    # if type(cat_cat_matrix_df) == type(None):
    location = f"plot/correlation_plot/cat_cat_heatmap.html"
    cat_cat_mat.write_html(file=location, default_width="40%", include_plotlyjs="cdn")
    correlation_plot.append(location)

    # cat cat msd table
    cat_cat_msd_df = msd.cat_cat_diff(X, y, cat_names, response, response_type)
    print("\nresult cat cat msd\n", cat_cat_msd_df)

    # ------------------------- Printing All matrices-----------------------------
    for i in correlation_matrices:
        print(i)

    # Html pages for correlation table with linked plots and msd table with plots
    # correlation table stored in plot/hw_4_plot with name of ""cr.html
    # msd table stored in plot/ with name brute_force "brute_force.html"
    dataframes = [cat_cat_table, cat_con_table, con_con_table]
    link.generate_html_cr(dataframes, path)
    dataframes_brute_force = [cat_cat_msd_df, cat_con_msd_df, con_con_msd_df]
    link.generate_html_brute_force(
        dataframes_brute_force, os.path.abspath("plot/brute_force")
    )

    # ------------------------- Modelling-----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=1234
    )

    pipe = Pipeline([("Normalize", Normalizer()), ("LDA", LDA(n_components=1))])
    pipe.fit(X_train, y_train)
    y_predict = pipe.predict(X_test)
    confusion_mat = confusion_matrix(y_test, y_predict)
    accuracy_LDA = accuracy_score(y_test, y_predict)
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
    print("Accuracy of RFC", accuracy_RFC)
    print(
        "Classification Report of Random Forest Classifier\n",
        classification_report(y_test, y_predict),
    )

    # Logistic Regression

    logistic = LogisticRegression(solver="lbfgs", max_iter=1000)
    logistic.fit(X_train, y_train)
    y_predict = logistic.predict(X_test)
    accuracy_LR = accuracy_score(y_test, y_predict)
    print("Accuracy of Logistic Regression", accuracy_LR)
    print(
        "Classification Report of Logistic Regression\n",
        classification_report(y_test, y_predict),
    )

    # support vector machine
    support_vector = svm.SVC()
    support_vector.fit(X_train, y_train)
    y_predict = support_vector.predict(X_test)
    accuracy_SV = accuracy_score(y_test, y_predict)
    print("Accuracy of SV", accuracy_SV)
    print(
        "Classification Report of Support Vector Classifier\n",
        classification_report(y_test, y_predict),
    )

    # NOTE: The accuracy of Support Vector Classifier model is slightly better
    # than Random forest, Logistic Regression and LDA.


if __name__ == "__main__":
    main()
