import os
import statistics

import numpy
import pandas as pd
import statsmodels.api
from flask import Flask, send_from_directory
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objs as go
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix

# Global variables for creating html file
tvalue = {}
pvalue = {}
plot = {}
model_plot = {}
weight_table = {}
unweight_table = {}
var_importance = {}
MeanSquareDiff_plot = {}
MeanSquareDiff_weighted = {}
MeanSquareDiff_unweighted = {}

app = Flask(__name__, static_url_path="")


@app.route("/plot/<path:path>")
def send_plot(path):
    return send_from_directory("plot", path)


@app.route("/")
def send_index():
    return send_from_directory(".", "Assignment.html")


# Function checking for boolean or continuous Response
def check_for_response(col):
    if len(set(col)) == 2:
        return True
    else:
        return False


# Function checking for boolean or continuous Predictor
def check_for_predictor(column):
    if column.dtypes == "object" or (column.nunique() / column.count() < 0.05):
        return True
    else:
        return False


# Logistic regression model for categorical response
def logistic_regression(y, predictor, feature):
    logistic_regression_model = statsmodels.api.Logit(y, predictor)
    logistic_regression_model_fitted = logistic_regression_model.fit()
    print(f"Feature_Name:{feature}")
    print(logistic_regression_model_fitted.summary())

    # Get the statistics
    tval = round(logistic_regression_model_fitted.tvalues[1], 6)
    pval = "{:.6e}".format(logistic_regression_model_fitted.pvalues[1])

    return tval, pval


# Linear regression model for continuous response
def linear_regression(y, predictor, column):
    linear_regression_model = statsmodels.api.OLS(y, predictor)
    linear_regression_model_fitted = linear_regression_model.fit()
    print(f"Feature_Name: {column}")
    print(linear_regression_model_fitted.summary())

    # Get statistics
    tval = round(linear_regression_model_fitted.tvalues[1], 6)
    pval = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

    return tval, pval


# Function for plotting Heatmap plot
def cat_response_cat_predictor(feature, column, Y):
    conf_matrix = confusion_matrix(feature, Y)

    fig = go.Figure(data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max()))
    fig.update_layout(
        title="Categorical Predictor by Categorical Response (with relationship)",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    # fig.show()
    file_name = f"plot/plot_{column}.html"
    fig.write_html(
        file=file_name,
        include_plotlyjs="cdn",
    )
    plot[column] = file_name


# Function for plotting distribution plot
def cat_response_con_predictor(dataset_df, column, Y):
    group_labels = ["0", "1"]
    label_0 = dataset_df[Y == 0][column]
    label_1 = dataset_df[Y == 1][column]
    hist_data = [label_0, label_1]
    colors = ["slategray", "magenta"]
    fig = ff.create_distplot(hist_data, group_labels, bin_size=0.5, colors=colors)
    fig.update_layout(
        title="Continuous Predictor by Categorical Response",
    )
    # fig.show()
    file_name = f"plot/plot_{column}.html"
    fig.write_html(
        file=file_name,
        include_plotlyjs="cdn",
    )
    plot[column] = file_name


# Function for plotting Heatmap plot
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
    file_name = f"plot/plot_{column}.html"
    fig.write_html(
        file=file_name,
        include_plotlyjs="cdn",
    )
    plot[column] = file_name


# Function for plotting scatter plot
def con_response_con_predictor(feature, column, Y, response):
    fig = px.scatter(feature, x=column, y=Y, trendline="ols")
    fig.update_layout(title="Continuous Response by Continuous Predictor")
    fig.update_xaxes(title=column)
    fig.update_yaxes(title="response")
    # fig.show()
    file_name = f"plot/plot_{column}.html"
    fig.write_html(
        file=file_name,
        include_plotlyjs="cdn",
    )
    plot[column] = file_name


def unweighted_table(feature, column, Y):
    # y=feature.to_list()
    number_bins = 10
    print(min(feature), max(feature))
    bin_range = max(feature) - min(feature)
    # print(range)
    bins = bin_range / number_bins
    y = Y.to_list()
    table = pd.DataFrame(
        columns=[
            "LowerBin",
            "UpperBin",
            "BinCenter",
            "BinCount",
            "BinMean",
            "PopulationMean",
            "MeanSquareDiff",
        ]
    )

    # mean square unweighted table
    for n in range(number_bins):
        low, high = min(feature) + (bins * n), min(feature) + (bins * (n + 1))
        feature_bin_list = []
        response_bin_list = []
        for i in range(len(feature)):

            # Below condition is to include high value in last bin
            if n == 9:
                if low <= feature[i] <= high:
                    feature_bin_list.append(feature[i])
                    response_bin_list.append(y[i])
            else:
                if low <= feature[i] < high:
                    feature_bin_list.append(feature[i])
                    response_bin_list.append(y[i])
        if not feature_bin_list:
            new_table = {
                "LowerBin": low,
                "UpperBin": high,
                "BinCenter": 0,
                "BinCount": 0,
                "BinMean": 0,
                "PopulationMean": numpy.nanmean(y),
                "MeanSquareDiff": 0,
            }
        else:
            bin_center = statistics.median(feature_bin_list)
            bin_count = int(len(feature_bin_list))
            bin_mean = statistics.mean(response_bin_list)
            pop_mean = numpy.nanmean(y)
            mean_sq_diff = round(abs((bin_mean - pop_mean) ** 2), 5)
            new_table = {
                "LowerBin": low,
                "UpperBin": high,
                "BinCenter": bin_center,
                "BinCount": bin_count,
                "BinMean": bin_mean,
                "PopulationMean": pop_mean,
                "MeanSquareDiff": mean_sq_diff,
            }
        table = table.append(new_table, ignore_index=True)
    return table


def weighted_table(Unweighted_Table, feature):
    population_proportion = Unweighted_Table["BinCount"] / len(feature)
    Unweighted_Table["PopulationProportion"] = population_proportion
    weighted_MSD = (
        Unweighted_Table["MeanSquareDiff"] * Unweighted_Table["PopulationProportion"]
    )
    Unweighted_Table["MeanSquaredDiffWeighted"] = weighted_MSD
    table = Unweighted_Table

    return table


def plot_weighted(Weighted_Table, feature, column):
    x = [i for i in Weighted_Table["BinCenter"]]
    y = [i for i in Weighted_Table["BinCount"]]
    print(column)
    plt = [
        go.Bar(x=x, y=y, yaxis="y2", name="population", opacity=0.4),
        go.Scatter(
            x=x,
            y=[i for i in Weighted_Table["BinMean"]],
            name="bin response mean",
            line=dict(color="red"),
        ),
        go.Scatter(
            x=[min(feature), max(feature)],
            y=[i for i in Weighted_Table["PopulationMean"]],
            name="population mean",
            line=dict(color="green"),
        ),
    ]
    layout = go.Layout(
        title="binned Difference with Mean of Response vs bin",
        xaxis_title=f"predictor: {column}",
        yaxis_title=f"response",
        yaxis2=dict(title="Population", overlaying="y", anchor="y3", side="right"),
    )
    fig = go.Figure(data=plt, layout=layout)
    file_name = f"plot/MSE_{column}.html"
    fig.write_html(
        file=file_name,
        include_plotlyjs="cdn",
    )
    MeanSquareDiff_plot[column] = file_name


def variable_importance(X, Y, response_type, predictors):
    if response_type is True:
        RandomForest = RandomForestClassifier()
        RandomForest.fit(X, Y)
        feature_importance = RandomForest.feature_importances_
    else:
        RandomForest = RandomForestRegressor()
        RandomForest.fit(X, Y)
        feature_importance = RandomForest.feature_importances_
    for var, imp in zip(predictors, feature_importance):
        var_importance[var] = round(imp, 5)


def main():

    if not os.path.exists("plot"):
        os.makedirs("plot")

    dataset = datasets.load_wine()
    dataset_df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
    dataset_df["target"] = dataset.target
    # print(dataset_df)

    # Identify predictors and response
    print(f"columns: {dataset_df.columns.to_list()}")

    # For Sklearn dataset, enter response as "target"
    response = input("Enter response column")

    # Predictors list containing all features
    predictors = [i for i in dataset_df if i != response]

    # X will print the data of all predictors and Y will print data of response
    X = dataset_df[predictors]
    Y = dataset_df[response]

    # Identify predictor and response type
    # predictors_type list containing True value for boolean data and False for continuous predictor
    # response_type variable containing True for boolean data and False for continuous response
    predictors_type = []
    response_type = check_for_response(Y)
    for predictor in predictors:
        predictors_type.append(check_for_predictor(dataset_df[predictor]))

    # Checking for each predictors_type and response_type
    for index, column in enumerate(predictors):
        feature = dataset_df[column]
        predictor = statsmodels.api.add_constant(feature)

        # Plotting graphs for each condition
        if predictors_type[index] is True and response_type is True:
            cat_response_cat_predictor(predictor, column, Y)
        elif predictors_type[index] is False and response_type is True:
            cat_response_con_predictor(dataset_df, column, Y)
        elif predictors_type[index] is True and response_type is False:
            con_response_cat_predictor(dataset_df, column, Y, response)
        else:
            con_response_con_predictor(predictor, column, Y, response)

        # Below condition true means predictor and response are continuous
        # linear regression model for calculating t value and p value
        if response_type is False:
            t_val, p_val = linear_regression(Y, predictor, column)
            print(f"tval:{t_val},\npval:{p_val}")
            tvalue[column] = t_val
            pvalue[column] = p_val

            fig = px.scatter(x=feature, y=Y, trendline="ols")
            fig.update_layout(
                title=f"Variable: {column}: (t-value={t_val}) (p-value={p_val})",
                xaxis_title=f"Variable:{column}",
                yaxis_title=response,
            )

            # writing plot on html file
            file_name = f"plot/model_plot_{column}.html"
            fig.write_html(file=file_name, include_plotlyjs="cdn")
            model_plot[column] = file_name

        # Below condition true means predictor is continuous and response is boolean
        # logistic regression to calculate p value and t value
        else:
            t_val, p_val = logistic_regression(Y, predictor, feature)
            print(f"t_value:{t_val} \np_value:{p_val}")
            tvalue[column] = t_val
            pvalue[column] = p_val

            fig = px.scatter(x=feature, y=Y, trendline="ols")
            fig.update_layout(
                title=f"Variable: {column}: (t-value={t_val}) (p-value={p_val})",
                xaxis_title=f"Variable: {column}",
                yaxis_title=response,
            )
            # fig.show()
            file_name = f"plot/model_plot_{column}.html"
            fig.write_html(file=file_name, include_plotlyjs="cdn")
            model_plot[column] = file_name

        # Generating Unweighted Table
        Unweighted_Table = unweighted_table(feature, column, Y)
        html_unweight = Unweighted_Table.to_html()
        file_name = f"plot/unweight_table_{column}.html"
        file = open(file_name, "w")
        file.write(html_unweight)
        file.close()
        unweight_table[column] = file_name
        MeanSquareDiff_unweighted[column] = (
            Unweighted_Table["MeanSquareDiff"].sum() / 10
        )

        # Generating Weighted Table
        Weighted_Table = weighted_table(Unweighted_Table, feature)
        html_weight = Weighted_Table.to_html()
        file_name = f"plot/weight_table_{column}.html"
        file = open(file_name, "w")
        file.write(html_weight)
        file.close()
        weight_table[column] = file_name
        MeanSquareDiff_weighted[column] = (
            Weighted_Table["MeanSquaredDiffWeighted"].sum() / 10
        )

        # plotting weighted graph
        plot_weighted(Weighted_Table, feature, column)

    # Random Forest variable Importance ranking
    variable_importance(X, Y, response_type, predictors)

    # Code for HTML page
    data = []
    sorted_importance = dict(
        sorted(var_importance.items(), key=lambda x: x[1], reverse=True)
    )
    print(sorted_importance)

    for column, value in sorted_importance.items():
        data.append(
            dict(
                response="response",
                predictor=column,
                Importance=sorted_importance[column],
                plots="http://localhost:5000/" + plot[column],
                model_plots="http://localhost:5000/" + model_plot[column],
                unweight="http://localhost:5000/" + unweight_table[column],
                MSD_Unweighted=MeanSquareDiff_unweighted[column],
                weight="http://localhost:5000/" + weight_table[column],
                MSD_Weighted=MeanSquareDiff_weighted[column],
                MSD_Plot="http://localhost:5000/" + MeanSquareDiff_plot[column],
            )
        )
    df = pd.DataFrame(data)
    # print(df)

    html = df.to_html(escape=True, render_links=True)
    text_file = open("Assignment.html", "w")
    text_file.write(html)
    text_file.close()


# comment below function to execute without flask
def flask_server():
    app.run(host="localhost", port=5000)


if __name__ == "__main__":
    main()
    # flask_server()
