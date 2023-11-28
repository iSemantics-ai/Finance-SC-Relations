import os
import json
from typing import Dict, Any, Tuple
from src.utils.s3_manage import put_json_obj
from src.relation_extraction.infer import infer_from_trained
from src.matcher.core import SimCSE_Matcher
from reporter import process_relations, match_companies
import awswrangler as wr
from src.glue.glue_etl import GlueETL
from src.utils.logs import get_logger

icon = "\U0001F4AB "
logger = get_logger(f"{icon}RE JOB", log_level="INFO")
############### Variables ################
CURRENT_STEP = "RE"
FOLLOWING = "Final"
distribute = True
##########################################
# Load GlueEtl Worker
etl = GlueETL()


# Reference it in the inference container at /opt/ml/model/code
def model_fn(model_dir: str) -> Tuple[infer_from_trained, SimCSE_Matcher]:
    """
    Loads the trained relation extractor and entity matcher models and returns them
    as a tuple.
    """
    relation_extractor = infer_from_trained(detect_entities=False)
    entity_matcher = SimCSE_Matcher(
        model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"
    )
    model_args = {"model_path": os.path.join(model_dir, "re_model"), "batch_size": 64}
    relation_extractor.load_model(model_args)
    return relation_extractor, entity_matcher


def float_format(x: Any) -> float:
    """
    Converts the given argument to a float and returns it.
    """
    return float(x)


def input_fn(request_body: str, content_type: str) -> Dict[str, Any]:
    """
    Parses the incoming request data and returns it as a dictionary.
    """
    if content_type == "application/json":
        request_content = json.loads(request_body)
    else:
        request_content = {}
    return request_content


def predict_fn(input_data: dict, model: tuple) -> dict:
    """
    Predicts relations between entities in the input data using a trained
    relation extraction model and an entity matching model.

    @params
    -------
    - input_data: A dictionary containing input data to be processed.
    - model: A tuple containing a trained relation extraction model and
            an entity matching model.

    @returns
    --------
    - A dictionary containing the processed input data.
    """
    # Print input data type and content
    print("Type of input:", type(input_data))
    print("Input data:", input_data)

    # Unpack the models
    relation_extractor, entity_matcher = model
    # Set the job input data
    etl.job = input_data

    # Update ETL starting status
    etl.update_starting(CURRENT_STEP, 1, add=False)

    # Block job files
    file_ids = etl.block_job_files(
        task=CURRENT_STEP,
        number_files=etl.config["job"][f"{CURRENT_STEP}_max_files"],
        distribute=distribute,
    )
    while file_ids:
        # Load data with filters
        query_data = etl.load_with_filter(
            etl.relations_table, col="accessionnumber", condition="isin", value=file_ids
        )

        # Filter for valid data
        valid_idx = query_data.query("supply_label == 1").index
        valid_data = query_data.iloc[valid_idx].copy()

        # Convert JSON strings to Python objects
        valid_data["spans"] = valid_data.spans.apply(json.loads)
        valid_data["org_groups"] = valid_data.org_groups.apply(json.loads)
        valid_data["aliases"] = valid_data.aliases.apply(json.loads)
        valid_data.reset_index(drop=True, inplace=True)
        # Set invalid files as successed
        success = set(query_data.accessionnumber.unique()) - set(
            valid_data.accessionnumber.unique()
        )

        # Predict relations
        predictions = relation_extractor.predict_relations(
            sentences=valid_data["sentence"].tolist(),
            ent="ORG",
            spans=valid_data["spans"].tolist(),
            org_groups=valid_data["org_groups"].tolist(),
            aliases=valid_data["aliases"].tolist(),
        )

        # Fill NaN values in predictions
        predictions.relations.fillna({}, inplace=True)
        valid_data.loc[predictions.index.values, "relations"] = None
        valid_data.loc[predictions.index.values, "relations"] = predictions["relations"]

        org_links = match_companies(
            predictions=valid_data,
            entity_matcher=entity_matcher,
            etl_worker=etl,
            lookup_table="company",
            index_column="companyprefix-normalizedname-index",
            attribute_name="companyprefix",
            prefix_len=2,
            sort_len=5,
            normalized_column="normalizedname",
            id_column="rgid",
            database_type="dynamodb",
            match_thresh=0.95,
            cand_thresh=0.80,
            top_k=5,
            index_memory="cuda",
        )

        relations_items = process_relations(valid_data, entity_matcher, org_links)
        failed_ingestions = etl.batch_write_dynamodb_items(
            relations_items, "Predictions"
        )
        # Create logs
        log_frame = etl.create_logs(query_data)

        if query_data.shape[0] > 0:
            # Ingest new data
            wr.s3.to_parquet(
                df=query_data,
                database=etl.database,
                table=etl.relations_table,
                dataset=True,
                path=etl.database_path + "/" + etl.relations_table,
                partition_cols=list(etl.relations_partitions.keys()),
                mode="overwrite_partitions",
                boto3_session=etl.session,
            )

            # Update logs
            wr.s3.to_parquet(
                df=log_frame,
                database=etl.database,
                table=etl.logs_table,
                dataset=True,
                path=etl.database_path + "/" + etl.logs_table,
                partition_cols=list(etl.logs_partitions.keys()),
                mode="overwrite_partitions",
                boto3_session=etl.session,
            )

            # Log success and failure
            logger.info(f"Ingested files with ids: {file_ids} successfully\u2705")
            success |= set(log_frame["accessionnumber"])
            failed = set(file_ids) - success
            update_response = etl.add_results(CURRENT_STEP, FOLLOWING, success, failed)

        else:
            logger.info("Didn't find any valid sentence for supply classification")

        # Block job files
        file_ids = etl.block_job_files(
            task=CURRENT_STEP,
            number_files=etl.config["job"][f"{CURRENT_STEP}_max_files"],
            distribute=distribute,
        )
    return input_data


def output_fn(prediction_output, accept="application/json"):
    return prediction_output
