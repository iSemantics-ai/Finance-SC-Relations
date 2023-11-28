import os
from typing import List, Dict
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
import re
import math
import numpy as np
import pandas as pd
import boto3
import awswrangler as wr
from src.utils.s3_utils import read_json_obj, put_json_obj, gen_date_condition
import os
import time
import boto3
import pandas as pd
import json
from utils.logs import get_logger
import boto3
from datetime import datetime
from tqdm.auto import tqdm


MODEL_INFERENCE = {
    "NER": "spacy_inference",
    "Supply-Chain-Classifier": "supply_detector_inference",
    "RE": "re_inference",
}
# ecomap-dl-pipeline
#'sagemaker-us-east-1-276066050088'
icon = "\U0001F30D"


class GlueETL:
    def __init__(self, required=set(["an", "date", "sens", "com"])):
        # Read Config file, If user arn not accessed to this object
        # the etl worker will not be valid
        s3 = boto3.resource("s3")
        obj = s3.Object(
            "sagemaker-us-east-1-276066050088", "sagemaker-jobs/config.json"
        )

        self.config = read_json_obj(obj)
        self.required = required
        self.logger = get_logger(f"{icon} ETL", log_level="INFO")
        # Define all required configurations
        self.database = self.config["database"]["name"]
        self.sentences_table = self.config["database"]["sentences_table"]
        self.relations_table = self.config["database"]["relations_table"]
        self.logs_table = self.config["database"]["logs_table"]
        self.bucket = self.config["database"]["bucket"]
        self.ref = self.config["database"]["ref"]
        self.region = self.config["base"]["region"]
        self.database_ref = self.config["database"]["database_ref"]
        self.database_path = f"s3://{self.bucket}/{self.database_ref}"
        self.queries_out = f"s3://{self.bucket}/queries/"
        self.input_path = self.config["database"]["input_path"]

        # Define all necessary clients and sessions.
        self.session = boto3.session.Session(
            region_name=self.region,
        )

        self.s3 = boto3.resource(
            "s3",
            region_name=self.region,
            
            
        )
        self.s3_client = boto3.client(
            "s3",
            region_name=self.region,
        )
        self.athena = boto3.client(
            "athena",
            region_name=self.region,
        )

        self.glue_client = boto3.client(
            "glue",
            region_name=self.region,
        )

        self.adaptor = self.s3.Object(
            self.bucket, f"{self.database_ref}/{self.config['database']['adaptor']}"
        )
        self.dynamodb = boto3.client(
            "dynamodb",
            region_name=self.region,
        )

        ## Read Definitions Schema
        schema_obj = self.s3.Object(
            "sagemaker-us-east-1-276066050088", "glue-db/database_schema.json"
        )
        self.definitions = read_json_obj(schema_obj)

        # Database Tables
        self.tables = {
            self.relations_table: (
                self.relations_definition,
                self.relations_comments,
                self.relations_description,
                self.relations_partitions,
            ),
            self.logs_table: (
                self.logs_definition,
                self.logs_comments,
                self.logs_description,
                self.logs_partitions,
            ),
        }

        for table, values in self.tables.items():
            if not wr.catalog.does_table_exist(
                database=self.database, table=table, boto3_session=self.session
            ):
                self.logger.info(
                    f"Creating `{table}` with definitions =\n{json.dumps(values[0], indent=2)} \nand partitions =\n {json.dumps(values[3],indent=2)}"
                )
                wr.catalog.create_parquet_table(
                    database=self.database,
                    table=table,
                    path=self.database_path + f"/{table}",
                    columns_types=values[0],
                    partitions_types=values[3],
                    compression="snappy",
                    description=values[2],
                    parameters={"source": "s3"},
                    columns_comments=values[1],
                    boto3_session=self.session,
                )

    @property
    def logs_definition(self):
        return self.definitions["logs_definition"]

    @property
    def relations_definition(self):
        return self.definitions["relations_definition"]

    @property
    def relations_comments(self):
        return self.definitions["relations_comments"]

    @property
    def logs_comments(self):
        return self.definitions["logs_comments"]

    @property
    def relations_description(self):
        return self.definitions["relations_description"]

    @property
    def logs_description(self):
        return self.definitions["logs_description"]

    @property
    def relations_partitions(self):
        return self.definitions["relations_partitions"]

    @property
    def logs_partitions(self):
        return self.definitions["logs_partitions"]

    @property
    def artifacts(self):
        obj = self.s3.Object(
            self.config["base"]["sagemaker_bucket"],
            "pipeline-artifacts/artifacts-config.json",
        )
        return obj

    @property
    def job(self):
        if not self._job_name:
            self.logger.info("No job specified!!!")
            return {}
        else:
            return self._job_name

    @property
    def job_attachment(self):
        message_obj = self.s3.Object(
            self.config["base"]["sagemaker_bucket"],
            f"jobs/{self._job_name}/sqs_message.json",
        )
        return read_json_obj(message_obj)

    @job.setter
    def job(self, message):
        """
        Sets the job name and creates necessary meta files for the job.

        @parmas
        -------
        - message (dict): The message containing the job information.
        """
        job_name = message["MessageTime"]
        bucket_name = self.config["base"]["sagemaker_bucket"]
        directory_name = f"jobs/{job_name}"
        response = self.s3_client.list_objects_v2(
            Bucket=bucket_name, Prefix=directory_name
        )
        if "Contents" in response:
            print(
                f"Directory '{directory_name}' exists in the S3 bucket '{bucket_name}'."
            )
            self._job_name = job_name
        else:
            # Create Meta JSON file
            meta_json = json.dumps({"freeze": False})
            models_meta = json.dumps(
                {
                    "n_inputs": 0,
                    "n_success": 0,
                    "n_failed": 0,
                    "n_blocked": 0,
                    "n_starting": 0,
                }
            )
            model_files = json.dumps(
                {"inputs": [], "success": [], "failed": [], "blocked": []}
            )
            models = list(MODEL_INFERENCE.keys())
            # Create all basic metafiles
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=directory_name + "/job_meta.json",
                Body=meta_json,
            )
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=directory_name + "/sqs_message.json",
                Body=json.dumps(message),
            )

            for model in models:
                self.s3_client.put_object(
                    Bucket=bucket_name,
                    Key=directory_name + f"/{model}/model_files.json",
                    Body=model_files,
                )
                self.s3_client.put_object(
                    Bucket=bucket_name,
                    Key=directory_name + f"/{model}/model_meta.json",
                    Body=models_meta,
                )
            self._job_name = job_name
            self.logger.info(f"Created new job with name: `{job_name}`")
            input_content = {"MessageTime": self._job_name, "TaskType": "Ingestion"}
            current_time = datetime.now()
            time_str = current_time.strftime("%Y-%m-%d-%H-%M-%S-%f")
            input_key = f"sqs_requests/{self._job_name}/Ingestion/{time_str}.json"
            input_obj = self.s3.Object(
                self.config["base"]["sagemaker_bucket"], input_key
            )
            put_json_obj(input_obj, input_content)
            self.logger.info("Running dl-flow...")

    @property
    def job_meta(self):
        if self._job_name:
            return self.s3.Object(
                self.config["base"]["sagemaker_bucket"],
                f"jobs/{self._job_name}/job_meta.json",
            )
        else:
            raise ("No job specified!!!")

    def model_meta(self, task):
        if self._job_name:
            return self.s3.Object(
                self.config["base"]["sagemaker_bucket"],
                f"jobs/{self._job_name}/{task}/model_meta.json",
            )
        else:
            raise ("No job specified!!!")

    def model_files(self, task):
        if self._job_name:
            return self.s3.Object(
                self.config["base"]["sagemaker_bucket"],
                f"jobs/{self._job_name}/{task}/model_files.json",
            )
        else:
            raise ("No job specified!!!")

    def freeze_job(self):
        if self._job_name:
            wait_count = 0
            while read_json_obj(self.job_meta)["freeze"] != False:
                time.sleep(5)
                wait_count += 1
                if wait_count > 6:
                    self.unfreeze_job()
            job_meta = read_json_obj(self.job_meta)
            job_meta["freeze"] = True
            put_json_obj(self.job_meta, job_meta)

        else:
            raise ("No job specified!!!")

    def unfreeze_job(self):
        if self._job_name:
            job_meta = read_json_obj(self.job_meta)
            job_meta["freeze"] = False
            put_json_obj(self.job_meta, job_meta)
        else:
            raise ("No job specified!!!")

    def block_job_files(self, task, number_files, distribute=True):
        """Block a specified number of input files for a given task.

        @paramas
        --------
        task (str): The name of the task.
        number_files (int): The number of input files to block.

        @returns
        --------
        List of files
        """
        # Count time consumed while running the function
        start_time = time.time()
        # Freeze the job to prevent new inputs from being added while blocking files
        self.freeze_job()
        # Get the model files and metadata objects for the task
        model_files_obj = self.model_files(task)
        model_meta_obj = self.model_meta(task)

        # Read the model metadata and get the number of input files
        model_meta = read_json_obj(model_meta_obj)

        n_inputs = model_meta["n_inputs"] - (
            model_meta.get("n_starting", 0) * number_files
        )
        # Calculate the number of input file transformations to block
        if n_inputs > 0:
            n_transformations = int(n_inputs / number_files) - 1
            number_of_files = number_files if n_transformations > 0 else n_inputs
            # Read the model files and pop the specified number of input files from the 'inputs' list
            model_files = read_json_obj(model_files_obj)
            files = model_files["inputs"][:number_of_files]
            model_files["inputs"] = model_files["inputs"][number_of_files:]
            # Add the blocked input files to the 'blocked' set in the model files
            model_files["blocked"] = list(set(model_files["blocked"]) | set(files))

            # Update the model metadata to reflect the blocked input files
            model_meta["n_inputs"] = len(model_files["inputs"])
            model_meta["n_blocked"] = len(model_files["blocked"])

            if n_transformations > 0 and distribute is True:
                self.logger.info("Request distribution task.")
                in_work = self.launch_batch_job(task_type="run-flow")
                model_meta_obj = self.model_meta(task)
                model_meta = read_json_obj(model_meta_obj)

            # Write the updated model files and metadata objects
            put_json_obj(model_files_obj, model_files)
            put_json_obj(model_meta_obj, model_meta)
            # Unfreeze the Job
            self.unfreeze_job()
            # Stop time counter
            end_time = time.time()
            time_consumed = end_time - start_time
            self.logger.info(
                f"Func(block_job_files)\nTime consumed to update the Job-Tracking files is {time_consumed} seconds"
            )
            return files
        # Unfreeze the Job
        self.unfreeze_job()
        return []

    def add_results(self, task: str, following: str, success: set, failed: set):
        """
        Update the Job-Tracking files with the results of a task.

        @params
        -------
        - task (str): the name of the task to update results for.
        - success (set): a set of file names that were successfully transformed.
        - failed (set): a set of file names that failed to be transformed.

        @returns
        --------
        - bool: True if the update was successful, False otherwise.

        """
        # Count time consumed while running the function
        start_time = time.time()

        # Freeze the job to prevent new inputs from being added while blocking files
        self.freeze_job()

        # Get the model files and metadata objects for the task
        model_files_obj = self.model_files(task)
        model_meta_obj = self.model_meta(task)

        # Read the model metadata and get the number of input files
        model_meta = read_json_obj(model_meta_obj)

        # Read the model files and pop the specified number of input files from the 'inputs' list
        model_files = read_json_obj(model_files_obj)

        # Record success transformations and add update following step tasks
        model_files["success"] = list(set(model_files["success"]) | success)
        model_meta["n_success"] = len(model_files["success"])

        # Record failed transformations
        model_files["failed"] = list(set(model_files["failed"]) | failed)
        model_meta["n_failed"] = len(model_files["failed"])

        # Record blocked updates
        model_files["blocked"] = list(set(model_files["blocked"]) - (success | failed))
        model_meta["n_blocked"] = len(model_files["blocked"])

        # Write the updated model files and metadata objects
        put_json_obj(model_files_obj, model_files)
        put_json_obj(model_meta_obj, model_meta)

        if not following == "Final":
            following_files_obj = self.model_files(following)
            following_meta_obj = self.model_meta(following)
            following_files = read_json_obj(following_files_obj)
            following_meta = read_json_obj(following_meta_obj)
            following_meta["n_inputs"] = len(model_files["success"])
            following_files["inputs"] = model_files["success"]
            put_json_obj(following_files_obj, following_files)
            put_json_obj(following_meta_obj, following_meta)
        # Unfreeze the Job
        self.unfreeze_job()

        # Stop time counter
        end_time = time.time()
        time_consumed = end_time - start_time

        # Log the time consumed by the function
        self.logger.info(
            f"Func(add_results)\nTime consumed to update the Job-Tracking files is {time_consumed} seconds"
        )

        return True

    def send_flow_message(self):
        # Create SQS client
        sqs = boto3.client(
            "sqs",
            region_name=self.region,
            
            
        )
        # Set the SQS queue URL
        queue_url = self.config["base"]["sqs_flow_url"]

        # Convert the message to JSON format
        message_body = json.dumps({"JobName": self._job_name, "TaskType": "run-flow"})
        # Send the message to the SQS queue
        response = sqs.send_message(QueueUrl=queue_url, MessageBody=message_body)
        return response

    def check_and_run(self, req_instances, task_type, model_name):
        input_content = {"MessageTime": self._job_name, "TaskType": task_type}
        running = 0
        # Check the number of running instances and subtract from the maximum instance limits
        running_instances = self.running_instances()
        limits = self.config["job"].get("instances")
        if len(running_instances) > 0:
            for k, v in running_instances.items():
                limits[k] -= v
        # Calculate the maximum number of instances that can be launched
        max_request = sum(limits.values())
        # Determine the number and type of instances to launch
        to_run = req_instances if req_instances <= max_request else max_request
        ready_instances = []
        for k, v in limits.items():
            ready_instances += [k] * v
        if len(ready_instances) == 0:
            # Add this job to the wait list
            wait_obj = self.s3.Object(
                self.config["base"]["sagemaker_bucket"],
                f"waitlist/{self._job_name}.json",
            )
            put_json_obj(wait_obj, input_content)
            self.logger.info(
                f"No instances available, adding task to the waitlist.\njob_name:{self._job_name}\ntask_type: {task_type}"
            )
        else:
            # Launch the instances and create input files for each instance
            for job_instances in ready_instances[:to_run]:
                current_time = datetime.now()
                time_str = current_time.strftime("%Y-%m-%d-%H-%M-%S-%f")
                input_key = f"job_trackes/{self._job_name}/{model_name}/{time_str}.json"
                output_key = f"sqs_requests/{self._job_name}/{model_name}"
                input_obj = self.s3.Object(
                    self.config["base"]["sagemaker_bucket"], input_key
                )
                put_json_obj(input_obj, input_content)

                # Start a batch transform job for the instance
                input_path = (
                    f"s3://{self.config['base']['sagemaker_bucket']}/{input_key}"
                )
                output_path = (
                    f"s3://{self.config['base']['sagemaker_bucket']}/{output_key}"
                )
                self.logger.info(
                    f"Running instance with \ntype: {job_instances}\ninput_path: {input_path}\noutput_path: {output_path}"
                )
                response = self.start_batch_transform_job(
                    model=model_name,
                    input_s3_uri=input_path,
                    output_s3_uri=output_path,
                    instance_type=job_instances,
                    instance_count=1,
                )
                running += 1

        return running

    def update_starting(self, model_name, running, add: bool):
        is_freeze = read_json_obj(self.job_meta)["freeze"]
        if not is_freeze:
            self.freeze_job()

        # Get the model metadata objects
        model_meta_obj = self.model_meta(model_name)
        # Read the model metadata and get the number of input files
        model_meta = read_json_obj(model_meta_obj)
        if add:
            model_meta["n_starting"] = model_meta["n_starting"] + running
            self.logger.info(
                f"Adding {running} instances as running for {model_name} task "
            )
        else:
            model_meta["n_starting"] = (
                model_meta["n_starting"] - 1 if model_meta["n_starting"] > 0 else 0
            )
            self.logger.info(
                f"Moving {running} instances to inProgress stage for {model_name} task "
            )

        put_json_obj(model_meta_obj, model_meta)
        if not is_freeze:
            self.unfreeze_job()

    def launch_batch_job(self, task_type):
        """
        Launch a batch transform job for each model that has input files.

        For each model that has input files, calculate the number of required instances based on the maximum number
        of input files per instance set in the configuration file. Check the number of currently running instances
        and subtract them from the maximum instance limits set in the configuration file. Launch the required number
        of instances with the available instance types. Create a new input file for each instance with a unique name
        based on the current date and time. Start a batch transform job for each instance with the corresponding input
        file and output path.

        Returns:
        - None
        """
        if task_type == "Ingestion":
            running = self.check_and_run(1, task_type, model_name="NER")
            self.update_starting("NER", 1, add=True)
        else:
            for model_name in MODEL_INFERENCE.keys():
                # Get the number of input files for the model
                n_inputs = read_json_obj(self.model_meta(model_name))["n_inputs"]
                if n_inputs > 0:
                    # Calculate the number of instances required to process the inputs
                    is_freeze = read_json_obj(self.job_meta)["freeze"]
                    if not is_freeze:
                        self.freeze_job()
                    # Get the model metadata objects
                    model_meta_obj = self.model_meta(model_name)
                    # Read the model metadata and get the number of input files
                    model_meta = read_json_obj(model_meta_obj)
                    n_starting = model_meta.get("n_starting", 0)
                    n_inputs = model_meta["n_inputs"] - (
                        n_starting * self.config["job"][f"{model_name}_max_files"]
                    )
                    req_instances = math.floor(
                        n_inputs / self.config["job"][f"{model_name}_max_files"]
                    )
                    if req_instances == 0 and n_starting == 0 and n_inputs > 0:
                        req_instances = 1
                    running = self.check_and_run(req_instances, task_type, model_name)
                    self.update_starting(model_name, running, add=True)
                    if not is_freeze:
                        self.freeze_job()
                else:
                    if model_name == "RE":
                        self.logger.info("Flow is over!!!")

    @classmethod
    def prepare_frame(cls, df, definition):
        definition_keys = np.array(list(definition.keys()))
        subset_mask = np.isin(definition_keys, df.columns)
        exists, non_exists = definition_keys[subset_mask], definition_keys[~subset_mask]
        df[non_exists] = None
        check_main = df[exists].isna().sum().all()
        check_estimations = df[non_exists].isna().sum().all()
        f"main exists: {not check_main}", f"estimations exists: {not check_estimations}"
        return df, exists, non_exists

    @classmethod
    def create_partition_logs(cls, df, definition):
        df, exists, non_exists = GlueETL.prepare_frame(df, definition)
        return pd.DataFrame(
            {
                "accessionnumber": [df["accessionnumber"].unique()[0]],
                "supply_estimations": [
                    df[["supply_label", "supply_score"]].notna().sum().min()
                ],
                "relation_estimations": [df[["relations"]].notna().sum().min()],
            }
        )


    def create_logs(self, data):
        """
        Given a DataFrame `data`, creates a summary of logs for each unique value
        in the "accessionnumber" column.

        Args:
            data (pandas.DataFrame): The input data to create logs from.

        Returns:
            pandas.DataFrame: A DataFrame summarizing the logs for each unique
            value in the "accessionnumber" column.

        """
        # Group the data by the "accessionnumber" column and count the number of rows in each group
        groups = data.groupby("accessionnumber")

        # Create a dictionary to hold the logs
        logs = {
            "accessionnumber": groups.size().index,
            "valid_for_supply": groups.apply(lambda x: (x["n_org"] > 1).sum()),
            "supply_estimations": groups.apply(lambda x: x["supply_label"].notna().sum()),
            "valid_for_re": groups.apply(lambda x: (x["supply_label"] == 1).sum()),
            "relation_estimations": groups.apply(lambda x: x["relations"].notna().sum()),
            "filedasofdate": groups["filedasofdate"].first(),
        }

        # Convert the logs dictionary to a DataFrame and return it
        return pd.DataFrame(logs).reset_index(drop=True)   
    
    def define_job_target(self, date):
        # load requested files
        requested_files = self.run_query(
            f"""SELECT DISTINCT accessionnumber 
         FROM {self.config['database']['source_db']}.{self.config['database']['source_table']}\
         WHERE filedasofdate like '{date}%'"""
        ).accessionnumber
        existed_files = self.run_query(
            f"""SELECT accessionnumber FROM {self.logs_table}\
         WHERE filedasofdate like '{date}%'"""
        ).accessionnumber
        valid_files = set(requested_files) - set(existed_files)
        # Freeze Job
        self.freeze_job()
        # Read CURRENT_STEP files
        # Define the Step Objects
        model_meta_obj = self.model_meta("NER")
        model_files_obj = self.model_files("NER")
        # Stream job_data
        model_meta = read_json_obj(model_meta_obj)
        model_files = read_json_obj(model_files_obj)
        # Ingest the new valid_files
        model_files["inputs"] = list(set(model_files["inputs"]) | set(valid_files))
        model_meta["n_inputs"] = len(model_files["inputs"])
        # Update the Job_data
        put_json_obj(model_files_obj, model_files)
        put_json_obj(model_meta_obj, model_meta)
        # Unfreeze the job
        self.unfreeze_job()
        # Return a tuple containing the list of valid file ids and the dictionary of failed files
        return {"valid_files": len(valid_files), "failed_files": 0}

    def running_instances(self):
        """Get the number of running instances for all active SageMaker batch transform jobs.

        Returns:
            dict: A dictionary where keys are instance types and values are the number of running instances.

        """
        # Create a Boto3 SageMaker client
        sm_client = boto3.client(
            "sagemaker",
            
            
            region_name=self.region,
        )

        # Initialize a dictionary to store the current number of running instances for each instance type
        current_jobs = defaultdict(lambda: 0)

        # List the batch transform jobs
        response = sm_client.list_transform_jobs()

        # Filter the results to show only the running jobs
        running_jobs = [
            job
            for job in response["TransformJobSummaries"]
            if job["TransformJobStatus"] == "InProgress"
        ]

        # Count the number of running instances for each instance type
        if len(running_jobs) > 0:
            for job in running_jobs:
                job_name = job["TransformJobName"]
                job_details = sm_client.describe_transform_job(
                    TransformJobName=job_name
                )
                instance_type = job_details["TransformResources"]["InstanceType"]
                instance_count = job_details["TransformResources"]["InstanceCount"]
                current_jobs[instance_type] += instance_count

        return current_jobs

    def ingest_from_bucket(self, bucket: str = None, prefix: str = "") -> None:
        """
        Ingests all JSON files located in a given S3 bucket and prefix into the AWS Glue Data Catalog.

        Args:
            bucket: str, the name of the S3 bucket containing the JSON files. This bucket must be in the same
                       AWS account and region as the AWS Glue Data Catalog.
            prefix: str, the prefix of the S3 object keys where the JSON files are located. For example, if the
                       S3 object keys are s3://my-bucket/data/json/file.json, the prefix would be 'data/json/'.
        Return:
            None
        """
        bucket_name = self.bucket if bucket is None else bucket
        s3 = boto3.client(
            "s3",
            region_name=self.region,
            
            
        )
        json_files = []
        # List all objects in the bucket
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        contents = response.get("Contents", [])
        if not contents:
            return [], [{prefix: {"Exception": "Invalid prefix"}}]
        # Loop through each object
        for content in contents:
            # Check if the object's key (file name) ends with .json
            if content["Key"].endswith(".json"):
                json_files.append(content["Key"])
        # If there are more than 1000 objects in the bucket, continue listing the objects
        while response["IsTruncated"]:
            continuation_key = response["NextContinuationToken"]
            response = s3.list_objects_v2(
                Bucket=bucket_name, ContinuationToken=continuation_key
            )
            for content in response["Contents"]:
                # Check if the object's key (file name) ends with .json
                if content["Key"].endswith(".json"):
                    json_files.append(content["Key"])
        # Map file paths into Pandas Frame to filer
        files_frame = pd.DataFrame({"file_path": json_files})
        # Extract file_ids from
        files_frame.loc[:, "file_id"] = [
            re.sub(".json", "", Path(filename).name) for filename in json_files
        ]

        files_frame.drop_duplicates("file_id", inplace=True)

        #
        try:
            is_exist = self.load_with_filter(
                table=self.logs_table,
                col="file_id",
                condition="isin",
                value=files_frame["file_id"].tolist(),
                columns=["file_id"],
            )["file_id"].tolist()
        except:
            is_exist = []
        # Filter out the pre-existed files
        files_frame = files_frame[~files_frame["file_id"].isin(is_exist)]
        if files_frame.shape[0] == 0:
            self.logger.info("Didn't detect any new JSON files")
            return [], []

        response = self.ingest_data(file_keys=files_frame["file_path"].tolist())
        return response

    def load_table(self, table_type="logs"):
        if table_type == "logs":
            return wr.s3.read_parquet_table(
                database=self.database,
                table=self.logs_table,
                use_threads=True,
                boto3_session=self.session,
            )

        return wr.s3.read_parquet_table(
            database=self.database,
            table=self.sentences_table,
            use_threads=True,
            boto3_session=self.session,
        )

    def get_tables_schema(self):
        """return all tables schemas"""
        response1 = self.glue_client.get_table(
            DatabaseName=self.database,
            Name=self.sentences_table,
        )
        response2 = self.glue_client.get_table(
            DatabaseName=self.database,
            Name=self.logs_table,
        )
        schemas = {
            self.sentences_table: response1["Table"]["StorageDescriptor"]["Columns"],
            self.logs_table: response2["Table"]["StorageDescriptor"]["Columns"],
        }

        return schemas

    def run_query(self, query_string: str, verbose: bool = True, r_type="df"):
        """Run athena query"""

        # submit the Athena query
        if verbose:
            print("Running query:\n " + query_string)

        query_execution = self.athena.start_query_execution(
            QueryString=query_string,
            QueryExecutionContext={"Database": self.database},
            ResultConfiguration={"OutputLocation": self.queries_out},
        )
        # wait for the Athena query to complete
        query_execution_id = query_execution["QueryExecutionId"]
        query_state = self.athena.get_query_execution(
            QueryExecutionId=query_execution_id
        )["QueryExecution"]["Status"]["State"]
        while query_state != "SUCCEEDED" and query_state != "FAILED":
            query_state = self.athena.get_query_execution(
                QueryExecutionId=query_execution_id
            )["QueryExecution"]["Status"]["State"]

        if query_state == "FAILED":
            failure_reason = self.athena.get_query_execution(
                QueryExecutionId=query_execution_id
            )["QueryExecution"]["Status"]["StateChangeReason"]
            print(failure_reason)
            df = pd.DataFrame()
            return df
        if r_type == "df":
            ## TODO: fix this to allow user-defined prefix
            results_file_prefix = f"queries/{query_execution_id}.csv"

            filename = f"{query_execution_id}.csv"
            try:
                if verbose:
                    print(f"self.bucket: {self.bucket}")
                    print(f"results_file_prefix: {results_file_prefix}")
                    print(f"filename: {filename}")

                self.s3.meta.client.download_file(
                    Bucket=self.bucket, Key=results_file_prefix, Filename=filename
                )
                df = pd.read_csv(filename)
                if verbose:
                    print(f"Query results shape: {df.shape}")
                os.remove(filename)
                self.s3.meta.client.delete_object(
                    Bucket=self.bucket, Key=results_file_prefix
                )
                self.s3.meta.client.delete_object(
                    Bucket=self.bucket, Key=results_file_prefix + ".metadata"
                )
                return df
            except Exception as inst:
                if verbose:
                    print(f"Failed download")
                    print(f"Exception: {inst}")
                df = None
                pass
        # in case want to return dict not dataf
        if r_type == "dict":
            return self.athena.get_query_results(QueryExecutionId=query_execution_id)

    def load_with_filter(
        self,
        table: str,
        database=None,
        col=None,
        condition=None,
        value=None,
        columns=None,
        lambda_fn=None,
    ):
        """query data with lambda mask

        filters = {
        '>=':lambda x: True if int(x[col]) >= value else False,
        '>':lambda x: True if int(x[col]) > value else False,
        '<=':lambda x: True if int(x[col]) <= value else False,
        '<':lambda x: True if int(x[col]) <= value else False,
        'isin':lambda x: True if x[col] in value else False,
        '==':lambda x: True if x[col] == value else False}
        """

        if lambda_fn is not None:
            filter_fn = lambda_fn
        else:
            filters = {
                ">=": lambda x: True if int(x[col]) >= value else False,
                ">": lambda x: True if int(x[col]) > value else False,
                "<=": lambda x: True if int(x[col]) <= value else False,
                "<": lambda x: True if int(x[col]) <= value else False,
                "isin": lambda x: True if x[col] in value else False,
                "==": lambda x: True if x[col] == value else False,
            }
            filter_fn = filters[condition]
        partition_data = wr.s3.read_parquet_table(
            database=self.database if not database else database,
            table=table,
            partition_filter=filter_fn,
            use_threads=True,
            columns=columns,
            boto3_session=self.session,
        )
        return partition_data

    def load_athena_with_filter(
        self,
        table,
        database=None,
        col=None,
        condition=None,
        value=None,
        custom_query=None,
        columns="*",
    ):
        """
        Performs a custom query on an Athena table, filtering the results based on the provided column, condition, and value.

        Args:
            col (str): The name of the column to filter on.
            condition (str): The filter condition to apply to the column.
            Examples include "=", ">", "<", ">=", "<=", "LIKE", and "IN".
            value (Optional[str]): The value to use in the filter condition. If condition is "IN",
            this should be a comma-separated list of values.
        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the rows that match the filter condition.
            Each dictionary represents a single row, with keys corresponding to column names and values corresponding to column values.

        Raises:
            ValueError: If the provided condition is not one of the supported conditions.
        """
        if not database:
            database = self.config["database"]["name"]
        query_filter = {
            "isin": f"""{col} IN ({",".join([f"'{f}'" for f in value] if isinstance(value,list) else [])})""",
            ">=": f"""{col} >= {value}""",
            ">": f"""{col} > {value}""",
            "<=": f"""{col} <= {value}""",
            "<": f"""{col} < {value}""",
            "==": f"""{col} = {value}""",
        }
        query_string = f"""SELECT {columns} FROM "{database}"."{table}" where {query_filter[condition]}"""
        query_data = self.run_query(query_string, r_type="df")
        return query_data

    def get_adaptor(self):
        """Read adaptor values"""
        self.logger.info("Read adaptor values")
        return read_json_obj(self.adaptor)

    def put_adaptor(self, content):
        self.logger.info("Adaptor Updated")
        return put_json_obj(self.adaptor, content)

    def reset_adaptor(self):
        """Reset all adaptor values to default"""
        json_content = read_json_obj(self.adaptor)
        json_content["wait"] = "false"
        json_content["ner_state"] = "active"
        json_content["supply_state"] = "active"
        json_content["supply_activated"] = []
        json_content["ner_activated"] = []
        json_content["re_activated"] = []
        self.logger.info("Reseting Adaptor Values to default")
        return put_json_obj(self.adaptor, json_content)

    def delete_glue_table_and_data(self, table_name=None):
        """
        Deletes an AWS Glue Data Catalog table and its associated S3 data.

        - table_name (str): The name of the table to delete.Default(None)
                if None the function will delete all the tables in the self.database

        Returns:
        - None.

        Raises:
        - Exception: An exception is raised if the Glue or S3 API calls fail.
        """
        # Get the S3 path of the table data
        if table_name is None:
            table_names = [self.relations_table, self.sentences_table, self.logs_table]
        else:
            table_names = [table_name]

        s3_client = boto3.client(
            "s3",
            region_name=self.region,
            
            
        )
        for table_name in table_names:
            try:
                table = self.glue_client.get_table(
                    DatabaseName=self.database, Name=table_name
                )
            except:
                self.logger.info(f"No Table founded with name `{table_name}`")
                continue

            s3_path = table["Table"]["StorageDescriptor"]["Location"]
            # Delete the table
            self.glue_client.delete_table(DatabaseName=self.database, Name=table_name)
            # Delete the S3 data
            bucket_name = s3_path.split("/")[2]
            prefix = "/".join(s3_path.split("/")[3:])
            try:
                response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
                objects = [{"Key": obj["Key"]} for obj in response["Contents"]]
                s3_client.delete_objects(
                    Bucket=bucket_name, Delete={"Objects": objects}
                )
            except:
                self.logger.info(
                    f"Failed to delete any associated data to {table_name}"
                )

    def update_artifacts(self, key_val: tuple):
        """
        Update the contents of a JSON file stored in an S3 bucket,
        which contains pipeline artifacts info.

        Args:
            key_val(tuple): this tuple must contains key-value pairs fot the artifact update

        Returns:
            None

        Raises:
            Exception: If there is an error updating the file in the S3 bucket.

        Example Usage:
            update_artifacts(('RE', 're_model'))
        """
        artifacts = read_json_obj(self.artifacts)
        if artifacts.get(key_val[0], False):
            artifacts.__setitem__(key_val[0], key_val[1])
            put_json_obj(self.artifacts, artifacts)

        else:
            raise ValueError(
                f"Invalid artifact with key <{key_val[0]}> , the avaliable keys is {artifacts.keys()}"
            )

    def create_transform_job_name(self, message_time, job_name):
        now = datetime.now()
        time_str = now.strftime("%H%M%S")
        transform_job_name = "-".join([job_name, message_time, time_str])
        return transform_job_name

    def create_sm_request(self, valid_files, message, job_name):
        sagemaker_bucket = self.config["base"]["sagemaker_bucket"]
        req_key = f'sqs_requests/{message["MessageTime"]}/dl-flow-{message["MessageTime"]}.json'
        input_s3_path = f"s3://{os.path.join(sagemaker_bucket, req_key)}"
        output_s3_path = f"s3://{os.path.join(sagemaker_bucket, 'sqs_requests')}"
        req_obj = self.s3.Object(sagemaker_bucket, req_key)
        message["FileIds"] = valid_files
        message["TaskType"] = "inference"
        message["Step"] = job_name
        # Put the batch transform input file
        put_request = put_json_obj(req_obj, message)
        return dict(
            put_request=put_request,
            input_s3_path=input_s3_path,
            output_s3_path=output_s3_path,
        )

    def start_batch_transform_job(
        self,
        model,
        input_s3_uri,
        output_s3_uri,
        instance_type="ml.g4dn.xlarge",
        instance_count=1,
    ):
        # Create a SageMaker client
        sagemaker = boto3.client(
            "sagemaker",
            region_name=self.region,
            
            
        )
        model_name = self.get_artifact(MODEL_INFERENCE.get(model))
        # Set up the input and output configurations
        input_data_config = [
            {
                "DataSource": {
                    "S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": input_s3_uri}
                }
            }
        ]
        output_data_config = {"S3OutputPath": output_s3_uri}
        # transform_job_name= self.create_transform_job_name(message_time, job_name)
        transform_job_name = input_s3_uri.split("/")[-1].replace(".json", "")
        # Set up the transform job parameters
        transform_params = {
            "TransformJobName": transform_job_name,
            "ModelName": model_name,
            "TransformInput": {
                "DataSource": {
                    "S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": input_s3_uri}
                },
                "SplitType": "Line",
                "ContentType": "application/json",
            },
            "TransformOutput": {"S3OutputPath": output_s3_uri},
            "TransformResources": {
                "InstanceType": instance_type,
                "InstanceCount": instance_count,
            },
        }
        # Start the batch transform job
        response = sagemaker.create_transform_job(**transform_params)
        return response

    def batch_write_dynamodb_items(
        self,
        dynamo_items: List[dict],
        table_name: str,
        dynamo_client: boto3.client = None,
    ) -> None:
        """
        Splits a list of DynamoDB items into batches of 25 and inserts them into a DynamoDB table using the batch_write_item() method.

        Args:
            dynamo_items (List[dict]): A list of DynamoDB items to be inserted into the table.
            table_name (str): The name of the DynamoDB table.
            dynamo_client (boto3.client): A Boto3 client object for communicating with DynamoDB.

        Returns:
            None: This function does not return anything, it simply inserts items into the table.

        Raises:
            None: This function does not raise any exceptions.

        """
        dynamo_client = dynamo_client or self.dynamodb
        # Split the items into batches of 25, the maximum batch size for DynamoDB
        batches = [dynamo_items[i : i + 25] for i in range(0, len(dynamo_items), 25)]

        # Insert each batch using the batch_write_item() method
        failed_batches = []
        for batch in tqdm(
            batches,
            total=len(batches),
            desc=f"Batch insertion into `{table_name}` Table",
        ):
            request_items = {
                table_name: [{"PutRequest": {"Item": item}} for item in batch]
            }
            try:
                response = dynamo_client.batch_write_item(RequestItems=request_items)
            except Exception as e:
                request_items["Exception"] = e
                failed_batches.append(request_items)
        print(f"Success rate {len(batches) - len(failed_batches)}/{len(batches)} ")
        return failed_batches

    def query_dynamodb_table(
        self,
        table_name: str,
        index_name: str,
        attribute_name: str,
        sort_name: str,
        query_values: List[str],
    ) -> List[Dict]:
        """
        Queries a DynamoDB table for a list of query values in the specified attribute.

        Args:
            table_name (str): The name of the DynamoDB table to query.
            index_name (str): The name of the secondary index to use for the query.
            attribute_name (str): The name of the attribute to search for the query values.
            query_values (List[str]): A list of values to query for in the specified attribute.
            dynamodb (boto3.client): A DynamoDB client instance.

        Returns:
            List[Dict]: A list of items that match the query values.
        """

        results = []
        key_condition_expression = (
            f"{attribute_name} = :val1 and begins_with({sort_name}, :val2)"
        )
        from tqdm import tqdm

        for query, sort_value in tqdm(query_values, total=len(query_values)):
            response = self.dynamodb.query(
                TableName=table_name,
                IndexName=index_name,
                KeyConditionExpression=key_condition_expression,
                ExpressionAttributeValues={
                    ":val1": {"S": query},
                    ":val2": {"S": sort_value},
                },
            )
            results.extend(response["Items"])

        return results

    def batch_scan(self, table_name, attribute_name, attribute_values, batch_size=150):
        """
        Performs a batched scan operation on a DynamoDB table and returns the query results as a list.

        :param table_name: The name of the DynamoDB table to scan.
        :param attribute_name: The name of the attribute to query.
        :param attribute_values: A list of attribute values to query.
        :param batch_size: The maximum number of items to query in each batch (default 150).
        :return: A list of query results.
        """
        # Calculate the number of batches needed
        step_number = math.ceil(len(attribute_values) / batch_size)
        index_name = "GS1"

        # Initialize the output list
        output = []

        # Loop through each batch
        for b_index in range(step_number):
            # Define start and end indices for the batch
            start = b_index * batch_size
            end = start + batch_size
            if b_index == step_number - 1:
                # If this is the last batch, include all remaining items
                batch = attribute_values[start:]
            else:
                # Otherwise, include the next batch of items
                batch = attribute_values[start:end]

            # Build the filter expression and expression attribute values for the batch
            filter_expression = " OR ".join(
                [f"{attribute_name} = :val{i}" for i in range(len(batch))]
            )
            expression_attribute_values = {
                f":val{i}": {"S": batch[i]} for i in range(len(batch))
            }
            # Perform the query for the batch
            response = self.dynamodb.scan(
                TableName=table_name,
                FilterExpression=filter_expression,
                ExpressionAttributeValues=expression_attribute_values,
            )

            # Append the query results to the output list
            output += response["Items"]
        return output

    def get_artifact(self, name: str):
        """
        Get value from the artifacts metadata

        Args:
            name(str): name of required key

        Returns:
            value of the key

        Raises:
            Exception: If key is not withing the metadata storage

        Example Usage:
            get_artifact('RE')
        """
        artifacts = read_json_obj(self.artifacts)
        response = artifacts.get(name, False)
        if response:
            return response
        else:
            raise ValueError(
                f"Invalid artifact with key <{name}> , the avaliable keys is {artifacts.keys()}"
            )

    def delete_unused_models(self):
        # Set the SageMaker client
        sm_client = boto3.client(
            "sagemaker",
            region_name=self.region,
            
            
        )
        s3_client = boto3.client(
            "s3",
            region_name=self.region,
            
            
        )
        inferences = [
            "ner_inference",
            "re_inference",
            "spacy_inference",
            "supply_detector_inference",
        ]
        infer_models = [self.get_artifact(name) for name in inferences]
        # List all the models
        models = sm_client.list_models()
        exists_models = [m["ModelName"] for m in models["Models"]]

        to_delete = list(set(exists_models) ^ set(infer_models))

        for model_name in to_delete:
            try:
                # Get the S3 location of the model artifacts
                response = sm_client.describe_model(ModelName=model_name)
                model_data_url = response["PrimaryContainer"]["ModelDataUrl"]
                s3_bucket = model_data_url.split("/")[2]
                s3_prefix = "/".join(model_data_url.split("/")[3:-1])

                # Delete the model
                sm_client.delete_model(ModelName=model_name)
                self.logger.info(f"Successfully deleted model with name: {model_name}")

                # Delete the S3 objects associated with the model
                s3_objects = s3_client.list_objects_v2(
                    Bucket=s3_bucket, Prefix=s3_prefix
                )["Contents"]
                # Delete the S3 objects associated with the model
                for obj in s3_objects:
                    s3_client.delete_object(Bucket=s3_bucket, Key=obj["Key"])
                    self.logger.info(
                        f"Successfully deleted object with key: {obj['Key']} in bucket {s3_bucket}"
                    )
                self.logger.info(f"Successfully deleted model with name: {model_name}")
            except:
                self.logger.info(f"Failed to deleted model with name: {model_name}")

    def drop_duplicates(self, table_name, database, columns=["file_id"]):
        """
        logs = wr.s3.read_parquet_table(
                                        database=database,
                                        table=table_name,
                                        use_threads = True,
                                        boto3_session=self.session
                                        )
        logs.drop_duplicates(columns, inplace=True)
        wr.s3.to_parquet(
                    df = logs,
                    database =database,
                    table = table_name,
                    dataset=True,
                    path=database_path+f'/{table_name}',
                    mode='overwrite',
                    sanitize_columns =True
                    )
        """
        pass

    def __str__(self):
        return f"""
Glue ETL Object INFO:
=====================
database = {self.database}
--------------------------
bucket = {self.bucket}
--------------------------
database_ref = {self.database_ref}
--------------------------
database_path = {self.database_path}
--------------------------
definitions = at `{f's3://{self.bucket}/{self.database_ref}/definitions.json'}`
--------------------------
sentences_table = {self.sentences_table}
--------------------------
logs_table = {self.logs_table}
                """
