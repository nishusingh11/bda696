"""
Code: To create mean square difference table weighted in descending order.
"""

import pandas as pd
from plotly import graph_objs as go


def con_con_diff(X, y, con_names, response, response_type):

    con_diff_plot = {}
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
    for p1 in con_names:
        con_diff_plot[p1] = {}
        for p2 in con_names:
            if p1 is not p2:
                b1 = pd.DataFrame(
                    {
                        "x1": X[p1],
                        "x2": X[p2],
                        "y": y,
                        "B1": pd.cut(X[p1], 10),
                        "B2": pd.cut(X[p2], 10),
                    }
                )
                b2 = (
                    b1.groupby(["B1", "B2"]).agg({"y": ["count", "mean"]}).reset_index()
                )
                b2.columns = [p1, p2, "bin_count", "bin_mean"]
                pp = b2.bin_count / len(y)
                b2["unweighted_msd"] = ((b2["bin_mean"] - pop_mean) ** 2) / 10
                b2["weighted_msd"] = ((b2["bin_mean"] - pop_mean) ** 2) * pp
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

                d = b2.pivot(index=p1, columns=p2, values="weighted_msd")

                graph = go.Figure(data=[go.Surface(z=d.values)])
                graph.update_layout(
                    title=f"{p1} vs {p2} plot",
                    autosize=True,
                    scene=dict(xaxis_title=p2, yaxis_title=p1, zaxis_title="z"),
                )

                filename = f"plot/brute_force/brute_force_con_{p1}_{p2}.html"
                graph.write_html(
                    file=filename,
                    include_plotlyjs="cdn",
                )
                con_diff_plot[p1][p2] = filename

    # print(con2_diff_plot)
    return con_con_msd_df, con_diff_plot


def cat_con_diff(X, y, con_names, cat_names, response, response_type):
    cols = [
        "response",
        "predictor1",
        "predictor2",
        "unweighted_msd",
        "weighted_msd",
    ]
    cat_con_msd_df = pd.DataFrame(columns=cols)
    if response_type == "continuous":
        pop_mean = y.mean()
    else:
        pop_mean = y.astype("category").cat.codes.mean()
    for p1 in cat_names:
        for p2 in con_names:
            if p1 is not p2:
                b1 = pd.DataFrame(
                    {
                        "x1": X[p1],
                        "x2": X[p2],
                        "y": y,
                        "B": pd.cut(X[p2].rank(method="first"), 10),
                    }
                )
                b2 = b1.groupby(["x1", "B"]).agg({"y": ["count", "mean"]}).reset_index()
                b2.columns = [p1, p2, "bin_count", "bin_mean"]
                pp = b2.bin_count / len(y)
                b2["unweighted_msd"] = ((b2["bin_mean"] - pop_mean) ** 2) / (
                    len(b2.bin_count)
                )
                b2["weighted_msd"] = ((b2["bin_mean"] - pop_mean) ** 2) * pp
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

                d = b2.pivot(index=p1, columns=p2, values="weighted_msd")
                graph = go.Figure(data=[go.Surface(z=d.values)])
                graph.update_layout(
                    title=f"{p1}vs {p2} Plot",
                    autosize=True,
                    scene=dict(xaxis_title=p2, yaxis_title=p1, zaxis_title="z"),
                )

                file = "plot/brute_force/brute_force_cat_con_" + p1 + "_" + p2 + ".html"
                graph.write_html(
                    file=file,
                    include_plotlyjs="cdn",
                )

    return cat_con_msd_df


def cat_cat_diff(X, y, cat_names, response, response_type):
    cols = [
        "response",
        "predictor1",
        "predictor2",
        "unweighted_msd",
        "weighted_msd",
    ]
    cat_cat_msd_df = pd.DataFrame(columns=cols)
    if response_type == "continuous":
        pop_mean = y.mean()
    else:
        pop_mean = y.astype("category").cat.codes.mean()
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
                b2["unweighted_msd"] = ((b2["bin_mean"] - pop_mean) ** 2) / (
                    len(b2.bin_count)
                )
                b2["weighted_msd"] = ((b2["bin_mean"] - pop_mean) ** 2) * pp
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
                d = b2.pivot(index=p1, columns=p2, values="weighted_msd")
                graph = go.Figure(data=[go.Surface(z=d.values)])
                graph.update_layout(
                    title=f"{p1} vs{p2} plot",
                    autosize=True,
                    scene=dict(xaxis_title=p2, yaxis_title=p1, zaxis_title="z"),
                )

                file = f"plot/brute_force/brute_force_cat_{p1}_{p2}.html"
                graph.write_html(
                    file=file,
                    include_plotlyjs="cdn",
                )

    return cat_cat_msd_df
