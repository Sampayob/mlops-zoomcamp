from batch import prepare_data, read_data


def test_read_data(input_data):
    """ """
    input_data.to_parquet("test_read_data.parquet")
    return_data = read_data("test_read_data.parquet")

    assert input_data.equals(return_data)


def test_prepare_data(input_data, expected_data):
    """ """
    categorical = ["PULocationID", "DOLocationID"]
    year = 2022
    month = 1

    assert expected_data.equals(prepare_data(input_data, categorical, month, year))
