import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import statsmodels.api
from plotly import express as px
import numpy as np
from plotly.subplots import make_subplots
from scipy import stats


# def diff(X, predictors, y, response, predictor_type):
#     weighted_mean_rank = {}
#     n_of_bin = 9
#     diff_plot = {}
#     BF_plot = {}
#     appended_data = []
#     MeanSquaredDiffWeighted = []
#     for column in predictors:
#         feature_name = column
#         predictor = X[column]
#         target = y
#         df = pd.DataFrame({feature_name: pd.Series(predictor)})
#         df["target"] = target
#         count_row = df.shape[0]
#         p_min = df[feature_name].min()
#         p_max = df[feature_name].max()
#         p_range = p_max - p_min
#         bin_width = p_range / n_of_bin
#         # to include min number
#         bin_list = [p_min - 1]
#         s = p_min
#         # +1 to include max number
#         while s < p_max + 1:
#             s += bin_width
#             bin_list.append(round(s, 0))
#
#         df_bin = df
#         df_bin["LowerBin_UpperBin"] = pd.cut(
#             x=df[feature_name],
#             bins=bin_list,
#             include_lowest=True,
#             duplicates="drop",  # noqa
#         )
#
#         bincenter = []
#         for bin_n in df_bin["LowerBin_UpperBin"]:
#             bincenter.append(bin_n.mid)
#
#             df_bin["BinCenters"] = pd.DataFrame(
#                 {"BinCenters": pd.Series(bincenter)}
#             )  # noqa
#             df_bin["response"] = df["target"]
#
#         df_bin["Name"] = pd.Series(np.repeat(feature_name, count_row))
#         df_bin["Type"] = pd.Series(np.repeat(predictor_type, count_row))
#
#         # Groupby df_bin table to create a Difference with mean table
#
#         df_response = df_bin.groupby(("LowerBin_UpperBin"), as_index=False)[
#             "response"
#         ].sum()
#
#         df_bin_groupby = df_bin.groupby(
#             ("LowerBin_UpperBin"), as_index=False
#         ).agg(  # noqa
#             bin_mean=pd.NamedAgg(column=feature_name, aggfunc="mean"),
#             bin_count=pd.NamedAgg(column=feature_name, aggfunc="count"),
#         )
#         df_bin_groupby["binned_response_mean"] = (
#                 df_response["response"] / df_bin_groupby["bin_count"]
#         )
#
#         bin_center_list = []
#         for bin_center in df_bin_groupby["LowerBin_UpperBin"]:
#             bin_center_list.append(bin_center.mid)
#
#         df_bin_groupby["BinCenter"] = pd.Series(bin_center_list)
#
#         PopulationMean = (np.sum(X[column])) / (count_row)
#         df_bin["PopulationMean"] = PopulationMean
#         df_bin_groupby["PopulationMean"] = PopulationMean
#
#         MeanSquaredDiff = (
#                                   df_bin_groupby["bin_mean"] - df_bin_groupby["PopulationMean"]
#                           ) ** 2
#         df_bin_groupby["MeanSquaredDiff"] = MeanSquaredDiff
#         weighted_mean_rank[column] = df_bin_groupby["MeanSquaredDiff"].sum()
#
#         # Square the difference, sum them up and divide by number of bins
#         #  print(
#         #      f"THE unWeighted NUMBER of {feature_name} IS : {df_bin_groupby['MeanSquaredDiff'].sum() / n_of_bin}"  # noqa
#         #  )
#         #  print(feature_name, df_bin_groupby)
#
#         trace1 = go.Bar(
#             x=df_bin_groupby["BinCenter"],
#             y=df_bin_groupby["bin_count"],
#             name="bin_count",
#             yaxis="y2",
#             opacity=0.5,
#         )
#         y2 = go.layout.YAxis(title="bin_count", overlaying="y", side="right")
#
#         trace2 = go.Scatter(
#             x=df_bin_groupby["BinCenter"],
#             y=df_bin_groupby["PopulationMean"],
#             name="population mean",
#             mode="lines",
#         )
#         trace3 = go.Scatter(
#             x=df_bin_groupby["BinCenter"],
#             y=df_bin_groupby["binned_response_mean"],
#             name="Bin Mean",
#         )
#
#         layout = go.Layout(
#             title="Binned Response Mean vs Population Mean",
#             xaxis_title=f"predictor: {feature_name}",
#             yaxis_title=f"Binned Response Mean",  # noqa
#             yaxis2=y2,
#         )
#
#         combined = [trace1, trace2, trace3]
#         fig = go.Figure(data=combined, layout=layout)
#         # fig.show()
#     return weighted_mean_rank
#
#
# def mean_diff_response(X, y, con_names, response, response_type):
#     weighted_table = []
#     weighted_mean_rank = {}
#     # DF=pd.DataFrame()
#     # DF["Response"]=response
#     # DF["Predictors"]=con_names
#     # print(DF)
#     for predictor in con_names:
#         mean_diff_df = pd.DataFrame({"intervals": pd.qcut(X[predictor], q=20)})
#         mean_diff_df[response] = y
#         mean_diff_df = mean_diff_df.groupby("intervals").agg(
#             {response: ["count", "mean"]}
#         )
#         mean_diff_df.columns = ["BinCount", "BinMean"]
#         mean_diff_df.reset_index(inplace=True)
#         mean_diff_df["PopMean"] = y.mean()
#         lower, upper = [], []
#         for interval in mean_diff_df.intervals:
#             lower.append(interval.left)
#             upper.append(interval.right)
#         mean_diff_df["LowerBin"] = lower
#         mean_diff_df["UpperBin"] = upper
#         mean_diff_df["BinCenter"] = (
#                                             mean_diff_df["LowerBin"] + mean_diff_df["UpperBin"]
#                                     ) / 2
#         mean_diff_df["MeanSquareDiff"] = round(abs((
#                                                            mean_diff_df["BinMean"] - mean_diff_df["PopMean"]
#                                                    ) ** 2), 5)
#         mean_diff_df["PopulationProp"] = mean_diff_df["BinCount"] / len(X)
#         mean_diff_df["MeanSquareDiffWeighted"] = (
#                 mean_diff_df["PopulationProp"] * mean_diff_df["MeanSquareDiff"]
#         )
#         weighted_mean_rank[predictor] = (round(mean_diff_df["MeanSquareDiffWeighted"].sum(), 5))
#         weighted_table.append(mean_diff_df)
#     print(weighted_table)
#     for table, predictor in zip(weighted_table, con_names):
#         mean_plot = make_subplots(specs=[[{"secondary_y": True}]])
#
#         mean_plot.add_trace(
#             go.Bar(
#                 x=table["BinCenter"],
#                 y=table["BinCount"],
#                 name="Hist",
#             ), secondary_y=False,
#         )
#         mean_plot.add_trace(
#             go.Scatter(
#                 x=table["BinCenter"],
#                 y=mean_diff_df["BinMean"] - mean_diff_df["PopMean"],
#                 name="Mean Difference",
#                 line=dict(color="red"),
#             ), secondary_y=True,
#         )
#         mean_plot.add_trace(
#             go.Scatter(
#                 x=table["BinCenter"],
#                 y=mean_diff_df["PopMean"],
#                 name="Population Mean",
#                 line=dict(color="green"),
#             ), secondary_y=True,
#         )
#         mean_plot.update_layout(
#             title=f"Predictor {predictor}",
#             xaxis_title="Bins",
#             yaxis_title="Response",
#         )
#         mean_plot.show()
#     return weighted_mean_rank


def new_diff(X, predictors, response, y):
    mean_diff_plot = {}
    rank = {}
    df = []
    unweight = []
    weight = []
    data1 = []
    table = pd.DataFrame(
        columns=[
            "BinCenter",
            "BinCount",
            "BinMean",
            "PopulationMean",
            "MeanSquareDiff",
        ]
    )

    n = 15
    response_list = y.tolist()
    for i in predictors:
        predictor_list = X[i].tolist()

        Count, _, _ = stats.binned_statistic(
            predictor_list, response_list, "count", bins=n
        )
        Mean, _, _ = stats.binned_statistic(
            predictor_list, response_list, "mean", bins=n
        )
        Center, _, _ = stats.binned_statistic(
            predictor_list, predictor_list, "median", bins=n
        )
        PopulationMean = sum(response_list) / len(response_list)
        Sd = (Mean - PopulationMean) ** 2
        Pp = Count / len(X[i])
        weighted_Sd = Sd * Pp

        unweighted_Msd = np.nansum(Sd) / n
        weighted_Msd = (np.nansum(weighted_Sd)) / n
        rank[i] = weighted_Msd
        table["BinCenter"] = Center
        table["BinCount"] = Count
        table["BinMean"] = Mean
        table["PopulationMean"] = PopulationMean
        table["MeanSquareDiff"] = weighted_Sd
        df.append(table)
        data1.append(
            dict(
                response="response",
                predictor=i,
                Unweighted_rank=unweighted_Msd,
                Weighted_rank=weighted_Msd,
            )
        )

        mean_plot = make_subplots(specs=[[{"secondary_y": True}]])

        mean_plot.add_trace(
            go.Bar(
                x=table["BinCenter"],
                y=table["BinCount"],
                name="Hist",
            ), secondary_y=False,
        )
        mean_plot.add_trace(
            go.Scatter(
                x=table["BinCenter"],
                y=table["BinMean"],
                name="Mean Difference",
                line=dict(color="red"),
            ), secondary_y=True,
        )
        mean_plot.add_trace(
            go.Scatter(
                x=table["BinCenter"],
                y=table["PopulationMean"],
                name="population Difference",
                line=dict(color="green"),
            ), secondary_y=True,
        )
        mean_plot.update_layout(
            title=f"Predictor {i}",
            xaxis_title="Bins",
            yaxis_title="Response",
        )
        mean_plot.update_layout(autosize=False, width=900, height=900)
        filename = f"plot/mean_diff/mean_diff_{i}.html"
        mean_plot.write_html(
            file=filename,
            include_plotlyjs="cdn",
        )
        mean_diff_plot[i] = filename
        # mean_plot.show()

    df2 = pd.DataFrame(data1)
    final_df = df2.sort_values(by=['Weighted_rank'], ascending=False)
    # print("sorted weighted rank\n", final_df)
    return rank, final_df
