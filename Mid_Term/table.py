"""
Code for correlation table
"""

import pandas as pd


def correlation_table(matrix, response):
    data = []
    if matrix:
        for i, v in matrix.items():
            for j in v:
                (
                    data.append(
                        dict(
                            response=response,
                            predictor1=i,
                            predictor2=j,
                            score=matrix[i][j],
                        )
                    )
                )
            # print("testing\n", data)
        result = pd.DataFrame(data)
        result = result.sort_values(by=["score"], ascending=False)
        return result
    else:
        return pd.DataFrame()
