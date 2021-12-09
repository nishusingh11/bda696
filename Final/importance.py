import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
import plotly as plt
import plotly.graph_objects as go


def variable_importance(X, Y, response_type, predictors):
    var_importance = {}
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
    return var_importance


def plot_feature_imp(importance, predictor):
    # feature_imp=np.array(importance)
    # feature_name= np.array(predictor)
    # df=pd.DataFrame(importance)
    feature_name = list(importance.keys())
    feature_imp = list(importance.values())

    # importance_df=pd.DataFrame(importance)
    # print(importance_df)
    feature_imp, feature_name = (list(i) for i in zip(*sorted(zip(feature_imp, feature_name))))
    plot_t = plt.subplots.make_subplots(rows=2, cols=1, vertical_spacing=0.35)
    plot_t.add_trace(go.Bar(name="importance", y=feature_imp, x=feature_name), row=1, col=1)
    plot_t.update_layout(title=go.layout.Title(text="Random Forest feature Importance", font=dict(
        family="Courier New, monospace",
        size=22,
        color="#0000FF"
    )))
    plot_t.update_layout(autosize=False, width=700, height=700)
    plot_t.update_xaxes(tickangle=25)
    plot_t.show()
