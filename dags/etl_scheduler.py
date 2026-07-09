from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator

# Import Databricks operator safely (in case it is not installed locally)
try:
    from airflow.providers.databricks.operators.databricks import (
        DatabricksSubmitRunOperator,
    )

    DATABRICKS_AVAILABLE = True
except ImportError:
    DATABRICKS_AVAILABLE = False

default_args = {
    "owner": "ai-engineer",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Determine the execution environment (local, databricks, etc.)
# Airflow Variable 'ENVIRONMENT' is used to toggle the execution mode.
environment = Variable.get("ENVIRONMENT", default_var="local").lower()
amazon_categories = Variable.get("SMARTSHOP_AMAZON_CATEGORIES", default_var="all")

with DAG(
    "smartshop_daily_etl",
    default_args=default_args,
    description="Daily ingestion & cleaning pipeline for Amazon products",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2026, 7, 1),
    catchup=False,
) as dag:

    if environment == "databricks" and DATABRICKS_AVAILABLE:
        # Retrieve Databricks settings from Airflow Variables
        cluster_id = Variable.get(
            "DATABRICKS_CLUSTER_ID", default_var="0706-spark-cluster-id"
        )
        script_path = Variable.get(
            "DATABRICKS_SCRIPT_PATH", default_var="dbfs:/scripts/spark_etl.py"
        )
        products_input = Variable.get(
            "SMARTSHOP_PRODUCTS_INPUT",
            default_var="dbfs:/mnt/raw-data/amazon_reviews_2023/combined/meta.jsonl",
        )
        reviews_input = Variable.get(
            "SMARTSHOP_REVIEWS_INPUT",
            default_var="dbfs:/mnt/raw-data/amazon_reviews_2023/combined/reviews.jsonl",
        )
        products_output = Variable.get(
            "SMARTSHOP_PRODUCTS_OUTPUT",
            default_var="dbfs:/mnt/processed-data/amazon_reviews_2023_flow_smoke",
        )
        output_format = Variable.get("SMARTSHOP_OUTPUT_FORMAT", default_var="delta")

        # Trigger Spark job on Databricks Cluster
        run_spark_etl = DatabricksSubmitRunOperator(
            task_id="run_spark_etl_job_databricks",
            existing_cluster_id=cluster_id,
            spark_python_task={
                "python_file": script_path,
                "parameters": [
                    "--input-products",
                    products_input,
                    "--input-reviews",
                    reviews_input,
                    "--output-path",
                    products_output,
                    "--output-format",
                    output_format,
                ],
            },
        )
    else:
        # Fallback to Local execution using BashOperator
        # This runs Spark in local mode on the Airflow worker
        workspace_path = Variable.get(
            "WORKSPACE_PATH",
            default_var=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        products_input = Variable.get(
            "SMARTSHOP_PRODUCTS_INPUT",
            default_var=(
                f"{workspace_path}/data/raw/amazon_reviews_2023/combined/meta.jsonl"
            ),
        )
        reviews_input = Variable.get(
            "SMARTSHOP_REVIEWS_INPUT",
            default_var=(
                f"{workspace_path}/data/raw/amazon_reviews_2023/combined/reviews.jsonl"
            ),
        )
        products_output = Variable.get(
            "SMARTSHOP_PRODUCTS_OUTPUT",
            default_var=f"{workspace_path}/data/processed/amazon_reviews_2023_flow_smoke",
        )
        output_format = Variable.get("SMARTSHOP_OUTPUT_FORMAT", default_var="parquet")
        max_products = Variable.get("SMARTSHOP_MAX_PRODUCTS", default_var="5000")
        max_reviews = Variable.get("SMARTSHOP_MAX_REVIEWS", default_var="20000")

        materialize_amazon_reviews = BashOperator(
            task_id="materialize_amazon_reviews_2023_local",
            bash_command=(
                f"python {workspace_path}/jobs/amazon_reviews_2023.py "
                f"--categories {amazon_categories} "
                f"--output-dir {workspace_path}/data/raw/amazon_reviews_2023 "
                f"--max-products {max_products} "
                f"--max-reviews {max_reviews}"
            ),
        )

        run_spark_etl = BashOperator(
            task_id="run_spark_etl_job_local",
            bash_command=(
                f"python {workspace_path}/jobs/spark_etl.py "
                f"--input-products {products_input} "
                f"--input-reviews {reviews_input} "
                f"--output-path {products_output} "
                f"--output-format {output_format} "
                f"--master 'local[*]'"
            ),
        )

        materialize_amazon_reviews >> run_spark_etl
