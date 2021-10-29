import pandas as pd
from plotly import graph_objs as go


def con_con_diff(X, y, con_names, response, response_type):
    # print("predictors\n", X)
    # print("con names are\n", con_names)
    con2_diff_plot = {}
    cols = [
        "response",
        "predictor1",
        "predictor2",
        "unweighted_msd",
        "weighted_msd",
    ]
    con_con_msd_df = pd.DataFrame(columns=cols)
    if response_type == "continuous":
        pop_mean = y.mean()
    else:
        pop_mean = y.astype("category").cat.codes.mean()
    # pop_mean = round(y.sum() / len(y), 5)
    # print("pp" ,pop_mean)
    for p1 in con_names:
        con2_diff_plot[p1] = {}
        for p2 in con_names:
            if p1 is not p2:
                b1 = pd.DataFrame(
                    {
                        "x1": X[p1],
                        "x2": X[p2],
                        "y": y,
                        "bucket1": pd.cut(X[p1].rank(method="first"), 10),
                        "bucket2": pd.cut(X[p2].rank(method="first"), 10),
                    }
                )
                b2 = (
                    b1.groupby(["bucket1", "bucket2"])
                    .agg({"y": ["count", "mean"]})
                    .reset_index()
                )
                b2.columns = [p1, p2, "bin_count", "bin_mean"]
                pp = b2.bin_count / len(y)
                b2["unweighted_msd"] = b2["bin_mean"] - pop_mean ** 2
                b2["weighted_msd"] = b2.unweighted_msd * pp
                con_con_msd_df = con_con_msd_df.append(
                    dict(
                        zip(
                            cols,
                            [
                                response,
                                p1,
                                p2,
                                b2["unweighted_msd"].sum(),
                                b2["weighted_msd"].sum(),
                            ],
                        )
                    ),
                    ignore_index=True,
                )
                con_con_msd_df = con_con_msd_df.sort_values(
                    by="weighted_msd",
                    ascending=False,
                )
                con_con_msd_df["weighted_msd"].round(5)
                d_mat = b2.pivot(index=p1, columns=p2, values="weighted_msd")

                fig = go.Figure(data=[go.Surface(z=d_mat.values)])
                fig.update_layout(
                    title=p1 + " " + p2 + "with bin difference weighted",
                    autosize=True,
                    scene=dict(xaxis_title=p2, yaxis_title=p1, zaxis_title="z"),
                )
                fig1 = go.Figure(data=go.Heatmap(z=d_mat.values, hoverongaps=False))
                fig1.update_xaxes(title_text=p1, showticklabels=False)
                fig1.update_yaxes(title=p2, showticklabels=False)
                fig1.update_layout(title=f"{p1}_{p2} weighted msd heatmap")
                # fig.show()
                # fig1.show()

                filename = f"plot/bf/bf_con_{p1}_{p2}.html"
                fig.write_html(
                    file=filename,
                    include_plotlyjs="cdn",
                )
                con2_diff_plot[p1][p2] = filename

    # print(con2_diff_plot)
    return con_con_msd_df, con2_diff_plot


def cat_con_diff(X, y, con_names, cat_names, response):
    cols = [
        "response",
        "predictor1",
        "predictor2",
        "unweighted_msd",
        "weighted_msd",
    ]
    cat_con_msd_df = pd.DataFrame(columns=cols)
    pop_mean = round(y.sum() / len(y), 5)
    for p1 in cat_names:
        for p2 in con_names:
            if p1 is not p2:
                b1 = pd.DataFrame(
                    {
                        "x1": X[p1],
                        "x2": X[p2],
                        "y": y,
                        "bucket": pd.cut(X[p2].rank(method="first"), 10),
                    }
                )
                b2 = (
                    b1.groupby(["x1", "bucket"])
                    .agg({"y": ["count", "mean"]})
                    .reset_index()
                )
                b2.columns = [p1, p2, "bin_count", "bin_mean"]
                pp = b2.bin_count / len(y)
                b2["unweighted_msd"] = (b2["bin_mean"] - pop_mean) ** 2
                b2["weighted_msd"] = b2.unweighted_msd * pp
                cat_con_msd_df = cat_con_msd_df.append(
                    dict(
                        zip(
                            cols,
                            [
                                response,
                                p1,
                                p2,
                                b2["unweighted_msd"].sum(),
                                b2["weighted_msd"].sum(),
                            ],
                        )
                    ),
                    ignore_index=True,
                )
                cat_con_msd_df = cat_con_msd_df.sort_values(
                    by="weighted_msd", ascending=False
                )

                d_mat = b2.pivot(index=p1, columns=p2, values="weighted_msd")
                fig = go.Figure(data=[go.Surface(z=d_mat.values)])
                fig.update_layout(
                    title=p1 + " " + p2 + " Plot",
                    autosize=True,
                    scene=dict(xaxis_title=p2, yaxis_title=p1, zaxis_title="target"),
                )

                filename = "plot/bf/bf_cat_con_" + p1 + "_" + p2 + ".html"
                fig.write_html(
                    file=filename,
                    include_plotlyjs="cdn",
                )

    return cat_con_msd_df


def cat_cat_diff(X, y, cat_names, response):
    cols = [
        "response",
        "predictor1",
        "predictor2",
        "unweighted_msd",
        "weighted_msd",
    ]
    cat_cat_msd_df = pd.DataFrame(columns=cols)
    pop_mean = round(y.sum() / len(y), 5)
    for p1 in cat_names:
        for p2 in cat_names:
            if p1 is not p2:
                b1 = pd.DataFrame(
                    {
                        "x1": X[p1],
                        "x2": X[p2],
                        "y": y,
                    }
                )
                b2 = (
                    b1.groupby(["x1", "x2"]).agg({"y": ["count", "mean"]}).reset_index()
                )
                b2.columns = [p1, p2, "bin_count", "bin_mean"]
                pp = b2.bin_count / len(y)
                b2["unweighted_msd"] = (b2["bin_mean"] - pop_mean) ** 2
                b2["weighted_msd"] = b2.unweighted_msd * pp
                cat_cat_msd_df = cat_cat_msd_df.append(
                    dict(
                        zip(
                            cols,
                            [
                                response,
                                p1,
                                p2,
                                b2["unweighted_msd"].sum(),
                                b2["weighted_msd"].sum(),
                            ],
                        )
                    ),
                    ignore_index=True,
                )
                cat_cat_msd_df = cat_cat_msd_df.sort_values(
                    by="weighted_msd", ascending=False
                )
                cat_cat_msd_df["weighted_msd"].round(5)
                d_mat = b2.pivot(index=p1, columns=p2, values="weighted_msd")
                # print("catcat\n", cat_cat_msd_df)
                fig = go.Figure(data=[go.Surface(z=d_mat.values)])
                fig.update_layout(
                    title=p1 + " " + p2 + " Plot",
                    autosize=True,
                    scene=dict(xaxis_title=p2, yaxis_title=p1, zaxis_title="target"),
                )

                filename = "plot/bf/bf_cat_" + p1 + "_" + p2 + ".html"
                fig.write_html(
                    file=filename,
                    include_plotlyjs="cdn",
                )

    return cat_cat_msd_df
