import os

import correlation_score
import data
import link
import msd
import numpy
import pandas as pd
import statsmodels.api
import table
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objs as go

plot = {}
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
    if column.dtypes.names in ["object", "category"] or (
        column.nunique() / column.count() < 0.05
    ):
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
    plot[column] = file_name


# Function for plotting distribution plot
def bool_response_con_predictor(dataset_df, response, column, Y):
    group_labels = ["0", "1"]
    label_0 = dataset_df[Y == 0][column]
    label_1 = dataset_df[Y == 1][column]
    hist_data = [label_0, label_1]
    colors = ["slategray", "magenta"]
    fig = ff.create_distplot(hist_data, group_labels, bin_size=10, colors=colors)
    fig.update_layout(
        title="Continuous Predictor by Boolean Response",
    )
    fig.show()
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

    """
    fig = px.histogram(
        dataset_df, x=column, histnorm="probability", barmode="overlay", marginal="rug",
        title="Continuous Predictor by Bool Response",
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
    """
    plot[column] = file_name


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
    plot[column] = file_name


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
    plot[column] = file_name


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
    logistic_regression_model = statsmodels.api.Logit(y, pred)
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
        # add title
        fig.update_layout(title_text=f"<i><b>{title}</b></i>")
        fig.show()
        return fig


def main():
    if not os.path.exists("plot/bf"):
        os.makedirs("plot/bf")
    if not os.path.exists("plot/correlation_plot"):
        os.makedirs("plot/correlation_plot")
    if not os.path.exists("plot/hw_4_plot"):
        os.makedirs("plot/hw_4_plot")
    path = os.path.abspath("plot/hw_4_plot")

    # Enter the Dataframe for datasets: "mpg", "tips", "titanic",
    # "boston", "diabetes", "breast_cancer", "Churn"
    df, predictor, response = data.get_test_data_set("Churn")

    X = df[predictor]
    y = df[response]

    # Identify Response Type
    response_type = check_for_response(y)

    if response_type == "boolean":
        df[response] = df[response].astype("category").cat.codes
    y = df[response]

    # Identify Predictor type
    predictor_type = [check_for_predictor(df[i]) for i in predictor]

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

    # print(df_cat_predictor.head(50))
    # print("cat_names", cat_names)
    # print(df_con_predictor)
    # print("con_names", con_names)

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
            # fig.show()
        elif response_type == "boolean" and predictor_type[index] == "continuous":
            t_val, p_val = logistic_regression(y, pred, column)
            fig = px.scatter(x=feature, y=y, trendline="ols")
            fig.update_layout(
                title=f"Variable: {column}: (t-value={t_val}) (p-value={p_val})",
                xaxis_title=f"Variable: {column}",
                yaxis_title=response,
            )
            # fig.show()

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

    con_con_table = table.correlation_table1(con_con_matrix, response)
    print("\ncon-con correlation table\n", con_con_table)

    # con-con matrix plot
    # con_mat = matrix_plot(con_con_table, "continuous continuous matrix plot")
    # con_mat.show()

    # saving con-con matrix in html file
    location = f"plot/correlation_plot/con_con_heatmap.html"
    con_mat.write_html(file=location, default_width="40%", include_plotlyjs="cdn")
    correlation_plot.append(location)

    # con con msd table
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

    cat_con_table = table.correlation_table2(cat_con_matrix, response)
    print("\ncat-con correlation table\n", cat_con_table)

    # Heatmap plot for cat-con matrix
    cat_con_plot = plot1(
        cat_con_matrix_d, "Correlation between Categorical and Continuous predictor"
    )
    # cat_con_mat.show()

    # saving cat-con matrix in html file
    location = f"plot/correlation_plot/cat_con_heatmap.html"
    cat_con_plot.write_html(file=location, include_plotlyjs="cdn")
    correlation_plot.append(location)

    # cat con msd table
    cat_con_msd_df = msd.cat_con_diff(X, y, con_names, cat_names, response)
    print("result cat con msd\n", cat_con_msd_df)

    # -----------correlation score, matrix and msd table for cat-cat----------------
    cat_cat_matrix = {}
    for i in cat_names:
        cat_cat_matrix[i] = {}
        for j in cat_names:
            cat_cat_matrix[i][j] = round(
                correlation_score.cat_correlation(X[i], X[j]), 5
            )

    # print(cat_cat_matrix)
    correlation_matrices.append(cat_cat_matrix)

    cat_cat_table = table.correlation_table3(cat_cat_matrix, response)
    print("\ncat-cat correlation table\n", cat_cat_table)

    cat_cat_matrix_df = pd.DataFrame(cat_cat_matrix)

    # Heatmap plot for cat-cat matrix
    cat_cat_mat = plot1(
        cat_cat_matrix_df, "Correlation between Categorical and Categorical predictor"
    )

    # saving cat-cat matrix in html file
    location = f"plot/correlation_plot/cat_cat_heatmap.html"
    cat_cat_mat.write_html(file=location, default_width="40%", include_plotlyjs="cdn")
    correlation_plot.append(location)

    # cat cat msd table
    cat_cat_msd_df = msd.cat_cat_diff(X, y, cat_names, response)
    print("\nresult cat cat msd\n", cat_cat_msd_df)

    # ------------------------- Printing All matrices-----------------------------
    for i in correlation_matrices:
        print(i)

    # Html pages for correlation table with linked plots and msd table with plots
    # correlation table stored in plot/hw_4_plot with name of ""cr.html
    # msd table stored in plot/bf with name bf "bf.html"
    dataframes = [cat_cat_table, cat_con_table, con_con_table]
    link.generate_html_cr(dataframes, path)
    dataframes_bf = [cat_cat_msd_df, cat_con_msd_df, con_con_msd_df]
    link.generate_html_bf(dataframes_bf, os.path.abspath("plot/bf"))


if __name__ == "__main__":
    main()
