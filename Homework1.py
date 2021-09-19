import pandas
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# Loading Iris dataset into dataframe using panda
def load(file_name):
    data = pandas.read_csv(file_name)
    data.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "class",
    ]

    print(data.head(10))
    print(data.info())
    return data


# Printing statistics summary using describe method.
def statistic_summary(data):
    # Summary Statistics
    print("\n", round(data.describe()), 3)


# Plotting 3D-Scatter, Violin, Scatter matrix,
# Histogram, Box plot using plotly
def plot_data(data):
    # 1. 3D-Scatter
    plot_3d_scatter = px.scatter_3d(
        data,
        x="sepal_length",
        y="sepal_width",
        z="petal_width",
        color="class",
        title="3-D Scatter plot",
    )

    plot_3d_scatter.show()

    # 2. violin
    plot_violin = px.violin(
        data, x="class", y="petal_length", color="class", box=True, title="Violin plot"
    )
    plot_violin.show()

    # 3. scatter matrix
    plot_scatter_matrix = px.scatter_matrix(
        data,
        dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"],
        color="class",
    )
    plot_scatter_matrix.show()

    # 4.histogram
    plot_histogram = px.histogram(
        data, x="petal_length", color="class", title="histogram"
    )
    plot_histogram.show()

    # 5. Box plot
    plot_box = px.box(data, color="class", title="Box Plot")
    plot_box.show()


# Analyse and Build model using StandardScaler() and RandomForestClassifier()
def random_forest(data):
    # Selecting features and target for Random Forest
    X = data.iloc[:, [0, 1, 2, 3]]
    Y = data.iloc[:, 4]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=1
    )
    pipeline = Pipeline(
        [
            ("standard_scale", StandardScaler()),
            ("random_forest", RandomForestClassifier(random_state=1234)),
        ]
    )
    pipeline.fit(X_train, Y_train)
    prediction = pipeline.predict(X_test)
    print("\n Random Forest Model Prediction\n")
    print(f"prediction:{prediction}")
    print(
        "Random Forest Accuracy\t{}".format(
            round(accuracy_score(Y_test, prediction), 5)
        )
    )


# Analyse and Build model using StandardScaler() and DecisionTreeClassifier()
def decision_tree(data):
    # Selecting Features and target for Decision Tree
    X = data.iloc[:, [0, 1, 2, 3]]
    Y = data.iloc[:, 4]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=1
    )
    pipeline = Pipeline(
        [
            ("standard_scale", StandardScaler()),
            ("decision_tree", DecisionTreeClassifier()),
        ]
    )
    pipeline.fit(X_train, Y_train)
    prediction = pipeline.predict(X_test)
    # Printing accuracy of Decision Tree Classifier
    print(
        "Decision Tree Accuracy\t{}".format(
            round(accuracy_score(Y_test, prediction), 5)
        )
    )


def main():
    data = load("iris.data")
    statistic_summary(data)
    plot_data(data)
    random_forest(data)
    decision_tree(data)


if __name__ == "__main__":
    main()
