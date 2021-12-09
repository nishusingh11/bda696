

import os

import correlation_score
import link
import msd
import numpy
import pandas as pd
import plotly.subplots
import sqlalchemy
import statsmodels.api
import table
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objs as go
from importance import variable_importance, plot_feature_imp
import M
import Models

# from pyspark import StorageLevel
# from pyspark.sql import SparkSession
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
p, t = {}, {}


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
        # title="Continuous Predictor by Boolean Response",
        title=column
    )
    fig.update_layout(autosize=False, width=700, height=700)
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
    pval = round(linear_regression_model_fitted.pvalues[1], 6)

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


def plot(p, t):
    p1, t1, p2, t2 = (
        list(p.values()),
        list(t.values()),
        list(p.keys()),
        list(t.keys()),
    )
    p1.sort()
    t1.sort()
    t1 = [abs(i) for i in t1]
    # print(p1)
    # print(t1)
    # print(p2)
    # print(t2)
    t1, t2 = (list(i) for i in zip(*sorted(zip(t1, t2))))
    p1, p2 = (list(i) for i in zip(*sorted(zip(p1, p2))))
    plot_t = plotly.subplots.make_subplots(rows=2, cols=1, vertical_spacing=0.35)
    plot_t.add_trace(go.Bar(name="tval", y=t1, x=t2), row=1, col=1)
    plot_t.update_layout(title=go.layout.Title(text="t-score plot", font=dict(
        family="Courier New, monospace",
        size=22,
        color="#0000FF"
    )))
    plot_t.update_layout(autosize=False, width=700, height=700)
    plot_t.update_xaxes(tickangle=45)
    plot_p = plotly.subplots.make_subplots(rows=2, cols=1, vertical_spacing=0.35)
    plot_p.add_trace(go.Bar(name="pval", y=p1, x=p2), row=1, col=1)
    plot_p.update_layout(title=go.layout.Title(text="p-score plot", font=dict(
        family="Courier New, monospace",
        size=22,
        color="#0000FF"
    )))
    plot_p.update_layout(autosize=False, width=700, height=700)
    plot_p.update_xaxes(tickangle=45)
    plot_t.show()
    plot_p.show()


def plot2(rank, title):
    rank_val, rank_key = (
        list(rank.values()),
        list(rank.keys()),
    )
    rank_val = [abs(i) for i in rank_val]
    # print(p1)
    # print(t1)
    # print(p2)
    # print(t2)
    rank_val, rank_key = (list(i) for i in zip(*sorted(zip(rank_val, rank_key))))
    # p1, p2 = (list(i) for i in zip(*sorted(zip(p1, p2))))
    plot_rank = plotly.subplots.make_subplots(rows=2, cols=1, vertical_spacing=0.35)
    plot_rank.add_trace(go.Bar(name="Weigthed Rank", y=rank_val, x=rank_key), row=1, col=1)
    plot_rank.update_layout(autosize=False, width=700, height=700)
    plot_rank.update_layout(title=go.layout.Title(text=title, font=dict(
        family="Courier New, monospace",
        size=22,
        color="#0000FF"
    )))
    plot_rank.update_xaxes(tickangle=45)
    print("rank")
    plot_rank.show()


def main():
    if not os.path.exists("plot/brute_force"):
        os.makedirs("plot/brute_force")
    if not os.path.exists("plot/correlation_plot"):
        os.makedirs("plot/correlation_plot")
    if not os.path.exists("plot/hw_4_plot"):
        os.makedirs("plot/hw_4_plot")
    if not os.path.exists("plot/mean_diff"):
        os.makedirs("plot/mean_diff")
    path = os.path.abspath("plot/hw_4_plot")
    if not os.path.exists("plot/tval_pval"):
        os.makedirs("plot/tval_pval")

    user = "root"
    password = "abc123"  # pragma: allowlist secret
    host = "localhost:3306"  # pragma: allowlist secret
    db = "baseball"
    c = f"mariadb+mariadbconnector://{user}:{password}@{host}/{db}"  # pragma: allowlist secret
    query = "SELECT * FROM feature_per"
    sql_engine = sqlalchemy.create_engine(c)
    df = pd.read_sql_query(query, sql_engine)
    # query.show()
    # df = dataframe.toPandas()

    df = df.drop(["home_team_id", "away_team_id", "game_id"], axis=1)

    for column in df:
        df[column].fillna(0, inplace=True)
    df = df.round(3)
    print(df.dtypes)
    df.dropna(axis=1)
    df.to_csv("./data.cvs")
    response = "home_team_wins"
    X = df.iloc[:, :-1]
    print(X.describe())
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
        if response_type == "boolean" and predictor_type[index] == "categorical":
            t_val, p_val = linear_regression(y, pred, column)
            t[column] = t_val
            p[column] = p_val
            fig = px.scatter(x=feature, y=y, trendline="ols")
            fig.update_layout(
                title=f"Variable: {column}: (t-value={t_val}) (p-value={p_val})",
                xaxis_title=f"Variable:{column}",
                yaxis_title=response,
            )
            # fig.show()
        elif response_type == "boolean" and predictor_type[index] == "continuous":
            t_val, p_val = logistic_regression(y, pred, column)
            t[column] = t_val
            p[column] = p_val
            fig = px.scatter(x=feature, y=y, trendline="ols")
            fig.update_layout(
                title=f"Variable: {column}: (t-value={t_val}) (p-value={p_val})",
                xaxis_title=f"Variable: {column}",
                yaxis_title=response,
            )
            filename = f"plot/tval_pval/tval_pval_{column}.html"
            fig.write_html(
                file=filename,
                include_plotlyjs="cdn",
            )
            # fig.show()
    plot(p, t)
    data3 = []
    for column, value in t.items():
        data3.append(
            dict(
                response="response",
                predictor=column,
                t_score=abs(value),
                p_score=p[column],
            )
        )
    df4 = pd.DataFrame(data3)
    df4 = df4.sort_values(by=["t_score"], ascending=False)
    print("t and p dataframe\n", df4)

    # Random Forest variable Importance ranking
    var_importance = variable_importance(X, y, response_type, predictor)
    sorted_importance = dict(
        sorted(var_importance.items(), key=lambda x: x[1], reverse=True)
    )
    print(sorted_importance)
    plot_feature_imp(sorted_importance, predictor)

    # Difference with Mean
    # msd.mean_diff_response(X, y, predictor, response, response_type)
    # msd.plot_weighted(weighted_table,predictor,)
    # rank=M.diff(X, predictor, y, response, predictor_type)
    # rank = M.mean_diff_response(X, y, con_names, response, response_type)
    rank, mean_diff_df = M.new_diff(X, predictor, response, y)
    plot2(rank, "Weighted mean squared diff ")
    # rank=M.diff(X, predictor, y, response, predictor_type)

    data = []
    for column, value in sorted_importance.items():
        data.append(
            dict(
                response="response",
                predictor=column,
                Importance=sorted_importance[column],
            )
        )
    df1 = pd.DataFrame(data)
    # print(df)

    html = df1.to_html(escape=True, render_links=True)
    text_file = open("Assignment.html", "w")
    text_file.write(html)
    text_file.close()

    sorted_rank = dict(
        sorted(rank.items(), key=lambda x: x[1], reverse=True)
    )
    print(sorted_rank)
    # data1 = []
    # for column, value in sorted_rank.items():
    #     data1.append(
    #         dict(
    #             response="response",
    #             predictor=column,
    #             MSD_Weighted_rank=sorted_rank[column],
    #         )
    #     )
    # df2 = pd.DataFrame(data1)
    # print("sorted weighted rank",df2)

    # html = df2.to_html(escape=True, render_links=True)
    # text_file = open("Assignment3.html", "w")
    # text_file.write(html)
    # text_file.close()

    # -----------correlation score, matrix and msd table for con-con----------------

    con_con_matrix = round(df_con_predictor.corr(method="pearson"), 2)
    # con_con_matrix = round(df.corr(method="pearson"),2)
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

    # ------------------------- Printing All matrices-----------------------------
    for i in correlation_matrices:
        print(i)

    # Html pages for correlation table with linked plots and msd table with plots
    # correlation table stored in plot/hw_4_plot with name of ""cr.html
    # msd table stored in plot/ with name brute_force "brute_force.html"
    dataframes = [con_con_table]
    dataframes_mean_diff = [mean_diff_df]
    dataframes_p_tvalue=[df4]
    link.generate_html_cr(dataframes, path)
    dataframes_brute_force = [con_con_msd_df]
    link.generate_html_brute_force(
        dataframes_brute_force, os.path.abspath("plot/brute_force")
    )

    link.generate_html_mean_diff(dataframes_mean_diff, os.path.abspath("plot/mean_diff"))
    link.generate_html_t_pval(dataframes_p_tvalue, os.path.abspath("plot/tval_pval"))

    Models.modelling(X, y)


if __name__ == "__main__":
    main()
