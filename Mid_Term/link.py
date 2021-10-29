import pandas as pd

table_categories = ["Category", "Category And Continuous", "Continuous"]
category_ids = ["cat", "cat_con", "con"]


def __get_row_for_cr(data_row, category, index):
    data_row = data_row[1]
    response = data_row.get(key="response")
    predictor1 = data_row.get(key="predictor1")
    predictor2 = data_row.get(key="predictor2")
    score = str(data_row.get(key="score"))
    label_str = (
        "Response: <b>"
        + response
        + "</b> Predictor: <b>"
        + predictor1
        + "</b> Score: <b>"
        + score
        + "</b>"
    )
    predictor1_plot_file_str = "cr_" + category + "_" + predictor1 + ".html"
    predictor2_plot_file_str = "cr_" + category + "_" + predictor2 + ".html"
    row_str = (
        """
                    <tr>
                        <th> """
        + index
        + """ </th>
                        <td>
                        """
        + response
        + """
                        </td>
                        <td>
                            <button onclick="modalFunction(\'"""
        + label_str
        + "','"
        + predictor1_plot_file_str
        + """\')"
                            type="button" data-toggle="modal" data-target="#chartModal">"""
        + predictor1
        + """

                            </button>
                        </td>
                        <td>
                            <button onclick="modalFunction(\'"""
        + label_str
        + "','"
        + predictor2_plot_file_str
        + """\')"
                            type="button" data-toggle="modal" data-target="#chartModal">"""
        + predictor2
        + """

                            </button>
                        </td>
                        <td>"""
        + score
        + """
                        </td>
                    </tr>

    """
    )
    return row_str


def __get_row_for_bf(data_row, category, index):
    # data_row = data_row[0]
    # print("hjsjhsgyuqk\n",data_row)
    response = data_row[0]
    predictor1 = data_row[1]
    predictor2 = data_row[2]
    unweighted_msd = str(data_row[3])
    weighted_msd = str(data_row[4])
    if predictor1 == "NoneType":
        predictor1 = str(predictor1)
    if predictor2 == "NoneType":
        predictor1 = str(predictor2)
    plot = category + "_" + predictor1 + "_" + predictor2

    label_str = (
        "Response: <b>"
        + response
        + "</b> Predictor1: <b>"
        + predictor1
        + "</b> Predictor2: <b>"
        + predictor2
        + "</b> unweighted_msd: <b>"
        + unweighted_msd
        + "</b> weighted_msd: <b>"
        + weighted_msd
    )
    plot_file_name = "bf_" + plot + ".html"
    row_str = (
        """
                    <tr>
                        <th> """
        + index
        + """ </th>
                        <td>
                        """
        + response
        + """
                        </td>
                        <td>"""
        + predictor1
        + """
                        </td>
                        <td>"""
        + predictor2
        + """
                        </td>
                        <td>"""
        + unweighted_msd
        + """
                        </td>
                        <td>"""
        + weighted_msd
        + """
                        </td>
                        <td>
                            <button onclick="modalFunction(\'"""
        + label_str
        + "','"
        + plot_file_name
        + """\')"
                            type="button" data-toggle="modal" data-target="#chartModal">"""
        + plot
        + """

                            </button>
                        </td>
                    </tr>

    """
    )
    return row_str


def __get_initial_html_str():
    html_str_initial = """
                    <!DOCTYPE html>
                    <html>
                    <head>
                    <style>
                    table, th, td {
                      border: 1px solid black;
                    }
                    </style>
                    </head>
                    <body>

    <script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.1/jquery.min.js">
    </script> <link
    href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.1/css/bootstrap.min.css" rel="stylesheet"
    integrity="sha384-WskhaSGFgHYWDcbwN70/dfYBj47jz9qbsMId/iRN3ewGhXQFZCSftd1LZCfmhktB"
    crossorigin="anonymous"> <script
    src="https://stackpath.bootstrapcdn.com/bootstrap/4.1.1/js/bootstrap.min.js"
    integrity="sha384-smHYKdLADwkXOn1EmN1qk/HfnUcbVRZyYmZ4qpPea6sjB/pTJ0euyQp0Mk8ck+5T"
    crossorigin="anonymous"></script>
    <table>

                    """
    return html_str_initial


def __get_last_html_str():
    html_str_last = """
                </table>
                <script> function modalFunction(label_str, file_str) {
                  document.getElementById("chartModalLabel").innerHTML = label_str
                  document.getElementById('chartIFrame').src = file_str }
                </script>


                <div class="modal fade" style="left: -500px;" id="chartModal" tabindex="-1"
                role="dialog" aria-labelledby="chartModalLabel"
                aria-hidden="true"> <div class="modal-dialog" role="document">
                <div class="modal-content"
                style="width:200%; height: 500;"> <div class="modal-header"> <h5 class="modal-title"
                id="chartModalLabel"></h5> <button type="button" class="close" data-dismiss="modal"
                aria-label="Close"> <span aria-hidden="true">&times;</span>
                </button> </div> <div class="modal-body">
                <iframe id="chartIFrame" width="100%" height="500" src="">
                </div> </div></div></div></body>
                """
    return html_str_last


def __get_anchor_str(current_table_id, table_label):
    anchor_str = "GoTo: "
    for table_id in category_ids:
        if current_table_id == table_id:
            continue
        href_str = (
            "<a href='#"
            + table_id
            + "'>"
            + table_categories[category_ids.index(table_id)]
            + "</a> &nbsp;&nbsp;&nbsp;&nbsp;"
        )
        anchor_str = anchor_str + href_str
    return anchor_str


def __generate_table_cr(dataframe, table_label, table_id):
    anchor_str = __get_anchor_str(table_id, table_label)
    table_str = (
        anchor_str
        + """
                <h2 id=\'"""
        + table_id
        + "'>"
        + table_label
        + """ </h2>
                <table>
                    <tr>
                            <th></th>
                            <th>Response</th>
                            <th>Predictor1</th>
                            <th>Predictor2</th>
                            <th>Score</th>
                    </tr>
                """
    )
    i = 1
    for row in dataframe.iterrows():
        row_str = __get_row_for_cr(row, table_id, str(i))
        i = i + 1
        table_str = table_str + row_str

    table_str = (
        table_str
        + """
                                </table>
                                <br>
                                <br>
                                <br>
                                <br>
                            """
    )
    return table_str


def __generate_table_bf(dataframe, table_label, table_id):
    anchor_str = __get_anchor_str(table_id, table_label)
    table_str = (
        anchor_str
        + """
                <h2 id=\'"""
        + table_id
        + "'>"
        + table_label
        + """ </h2>
                <table>
                    <tr>
                            <th></th>
                            <th>Response</th>
                            <th>Predictor1</th>
                            <th>Predictor2</th>
                            <th>Unweighted_MSD</th>
                            <th>Weighted_MSD</th>
                            <th>Plot</th>
                    </tr>
                """
    )
    i = 1
    for index, row in dataframe.iterrows():
        # print("complete dataframe\n",dataframe)
        # print("row\n",row)
        row_str = __get_row_for_bf(row, table_id, str(i))
        i = i + 1
        table_str = table_str + row_str

    table_str = (
        table_str
        + """
                                </table>
                                <br>
                                <br>
                                <br>
                                <br>
                            """
    )
    return table_str


def generate_html_cr(dataframes, path):
    html_str = __get_initial_html_str()
    for i in range(0, len(table_categories)):
        html_str = html_str + __generate_table_cr(
            dataframes[i], table_categories[i], category_ids[i]
        )
    html_str = html_str + __get_last_html_str()
    hs = open(path + "/cr.html", "w")
    hs.write(html_str)


def generate_html_bf(dataframes, path):
    html_str = __get_initial_html_str()
    for i in range(0, len(table_categories)):
        html_str = html_str + __generate_table_bf(
            dataframes[i], table_categories[i], category_ids[i]
        )
    html_str = html_str + __get_last_html_str()
    hs = open(path + "/bf.html", "w")
    hs.write(html_str)

