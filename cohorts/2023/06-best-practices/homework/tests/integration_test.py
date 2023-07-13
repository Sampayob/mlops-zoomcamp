import os


def test_write_file_to_s3(df_input):
    year = 2022
    month = 2

    options = {
    'client_kwargs': {
        'endpoint_url':os.getenv('S3_ENDPOINT_URL')
        }
    }

    output_file = "s3://nyc-duration/out/{year:04d}-{month:02d}.parquet".format(year=year, month=month)

    df_input.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )