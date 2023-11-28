from pathlib import Path
import os.path
import json
import datetime
import awswrangler as wr
import tarfile

pandas_format = {".tsv": "csv", ".csv": "csv", ".xlsx": "excel", ".json": "json"}
pandas_params = {".tsv": {"sep": "\t"}, ".csv": {"index_col": 0}}


def read_file(bucket, file_key, session, drop=True):
    """Read s3 file and return dataframe

    Args:
    bucket (str): Bucket name
    file_key(str): The key of the file in s3
    session: boto3_session with required access
    drop (bool): Drop Duplicates Default(True)
    """
    df = None
    if not Path(file_key).suffix in [".tsv", ".csv", ".json", ".xlsx"]:
        return None

    file_id = Path(file_key).stem
    path = f"s3://{bucket}/{file_key}"
    suffix = Path(file_key).suffix
    df = wr.s3.__getattribute__(f"read_{pandas_format[suffix]}")(
        path=path, **pandas_params.get(suffix), boto3_session=session
    )

    df.columns = [col.lower() for col in df.columns]
    df.columns = ["_".join(col.split()) for col in df.columns]
    df["file_id"] = file_id

    if drop:
        df = df.drop_duplicates()
    return df


def read_json_obj(config_object):
    file_content = config_object.get()["Body"].read().decode()
    return json.loads(file_content)


def put_json_obj(config_object, json_content):
    response = config_object.put(Body=(bytes(json.dumps(json_content).encode("UTF-8"))))
    return response


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def get_current_time(add_minutes=0, add_seconds=0):
    now = datetime.datetime.now()
    new_time = now + datetime.timedelta(minutes=add_minutes, seconds=add_seconds)
    return new_time.strftime("%Y-%m-%d-%H-%M-%S")


def gen_date_condition(date):
    condition = ""
    for i, item in enumerate(date.split("-")):
        if i == 0:
            condition += f"year='{item}' "
        elif i == 1:
            condition += f"and month='{item}' "
        elif i == 2:
            condition += f"and day='{item}' "
    return condition
