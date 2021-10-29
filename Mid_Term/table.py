import pandas as pd


def correlation_table1(dictionary, response):
    data = []
    # result = pd.DataFrame(columns=["predictor1", "predictor2", "score"])
    for i, v in dictionary.items():
        for j in v:
            # if i != j:
            (
                data.append(
                    dict(
                        response=response,
                        predictor1=i,
                        predictor2=j,
                        score=dictionary[i][j],
                    )
                )
            )
    # print("testing\n", data)
    result = pd.DataFrame(data)
    result = result.sort_values(by=["score"], ascending=False)

    return result


def correlation_table2(dictionary, response):
    data = []
    # result = pd.DataFrame(columns=["predictor1", "predictor2", "score"])
    if dictionary:
        for i, v in dictionary.items():
            for j in v:
                # if i != j:
                (
                    data.append(
                        dict(
                            response=response,
                            predictor1=i,
                            predictor2=j,
                            score=dictionary[i][j],
                        )
                    )
                )
            # print("testing\n", data)
        result = pd.DataFrame(data)
        result = result.sort_values(by=["score"], ascending=False)
        return result
    else:
        return pd.DataFrame()


def correlation_table3(dictionary, response):
    data = []
    # result = pd.DataFrame(columns=["predictor1", "predictor2", "score"])
    for i, v in dictionary.items():
        for j in v:
            # if i != j:
            (
                data.append(
                    dict(
                        response=response,
                        predictor1=i,
                        predictor2=j,
                        score=dictionary[i][j],
                    )
                )
            )
    # print("testing\n",data)
    result = pd.DataFrame(data)
    result = result.sort_values(by=["score"], ascending=False)
    return result
