import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


def new_diff(X, predictors, y):
    mean_diff_plot = {}
    rank = {}
    df = []
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
            ),
            secondary_y=False,
        )
        mean_plot.add_trace(
            go.Scatter(
                x=table["BinCenter"],
                y=table["BinMean"],
                name="Mean Difference",
                line=dict(color="red"),
            ),
            secondary_y=True,
        )
        mean_plot.add_trace(
            go.Scatter(
                x=table["BinCenter"],
                y=table["PopulationMean"],
                name="population Difference",
                line=dict(color="green"),
            ),
            secondary_y=True,
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
    final_df = df2.sort_values(by=["Weighted_rank"], ascending=False)
    return rank, final_df
