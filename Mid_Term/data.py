import random
from typing import List

import pandas
import seaborn
from sklearn import datasets


def get_test_data_set(data_set_name: str = None) -> (pandas.DataFrame, List[str], str):
    """Function to load a few test data sets

    :param:
    data_set_name : string, optional
        Data set to load

    :return:
    data_set : :class:`pandas.DataFrame`
        Tabular data, possibly with some preprocessing applied.
    predictors :list[str]
        List of predictor variables
    response: str
        Response variable
    """
    d = ["Churn"]
    seaborn_data_sets = ["mpg", "tips", "titanic"]
    sklearn_data_sets = ["boston", "diabetes", "breast_cancer"]
    all_data_sets = seaborn_data_sets + sklearn_data_sets + d

    if data_set_name is None:
        data_set_name = random.choice(all_data_sets)
    else:
        if data_set_name not in all_data_sets:
            raise Exception(f"Data set choice not valid: {data_set_name}")

    if data_set_name in seaborn_data_sets:
        if data_set_name == "mpg":
            data_set = seaborn.load_dataset(name="mpg").dropna().reset_index()
            predictors = [
                "cylinders",
                "displacement",
                "horsepower",
                "weight",
                "acceleration",
                "origin",
            ]
            response = "mpg"
        elif data_set_name == "tips":
            data_set = seaborn.load_dataset(name="tips").dropna().reset_index()
            predictors = [
                "total_bill",
                "sex",
                "smoker",
                "day",
                "time",
                "size",
            ]
            response = "tip"
        elif data_set_name == "titanic":
            data_set = seaborn.load_dataset(name="titanic").dropna()
            predictors = [
                "pclass",
                "sex",
                "age",
                "sibsp",
                "embarked",
                "class",
            ]
            response = "survived"
    elif data_set_name in sklearn_data_sets:
        if data_set_name == "boston":
            data = datasets.load_boston()
            data_set = pandas.DataFrame(data.data, columns=data.feature_names)
            data_set["CHAS"] = pandas.Categorical(data_set["CHAS"])
        elif data_set_name == "diabetes":
            data = datasets.load_diabetes()
            data_set = pandas.DataFrame(data.data, columns=data.feature_names)
        elif data_set_name == "breast_cancer":
            data = datasets.load_breast_cancer()
            data_set = pandas.DataFrame(data.data, columns=data.feature_names)

        data_set["target"] = data.target
        predictors = data.feature_names
        response = "target"

    elif data_set_name == "Churn":
        data_set = pandas.read_csv("Mid_Term/Churn.csv")
        response = "Churn"
        # predictor = data_set.drop(['Churn'], axis=1)
        # df = pd.read_csv('Mid_Term/Churn.csv')
        data_set.dropna()
        # print(data.columns)
        # print(data.isna().any())
        # y = data_set["Churn"]
        X = data_set.drop(["Churn", "customerID"], axis=1)
        data_set["TotalCharges"] = data_set["TotalCharges"].replace(" ", 0)

        # convert to float64
        data_set["TotalCharges"] = data_set["TotalCharges"].astype("float64")
        predictors = X.columns.values

    return data_set, predictors, response


# def Churn():
#
#     data = pandas.read_csv(os.path.abspath('MidTerm/Churn.csv'))
#     response="Churn"
#     predictor=data.drop(['Churn'], axis = 1)
#     #df = pd.read_csv('Mid_Term/Churn.csv')
#     data.dropna()
#     # print(data.columns)
#     # print(data.isna().any())
#     y = data["Churn"]
#     X = data.drop(['Churn', 'customerID'], axis=1)
#     predictor = X.columns.values
#     print(data)
#     print(predictor)
#     print(response)
#
#     return data, predictor, response


if __name__ == "__main__":
    df, predictors, response = get_test_data_set()
    # data, predictor, response = Churn()
