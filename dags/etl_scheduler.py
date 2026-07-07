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

        # Trigger Spark job on Databricks Cluster
        run_spark_etl = DatabricksSubmitRunOperator(
            task_id="run_spark_etl_job_databricks",
            existing_cluster_id=cluster_id,
            spark_python_task={
                "python_file": script_path,
                "parameters": [
                    "--input-products",
                    "dbfs:/mnt/raw-data/amazon_products.jsonl",
                    "--input-reviews",
                    "dbfs:/mnt/raw-data/amazon_reviews.jsonl",
                    "--output-path",
                    "dbfs:/mnt/processed-data/products_delta",
                    "--output-format",
                    "delta",
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

        run_spark_etl = BashOperator(
            task_id="run_spark_etl_job_local",
            bash_command=(
                f"python {workspace_path}/jobs/spark_etl.py "
                f"--input-products {workspace_path}/data/raw/amazon_products.jsonl "
                f"--input-reviews {workspace_path}/data/raw/amazon_reviews.jsonl "
                f"--output-path {workspace_path}/data/processed/products_processed "
                f"--output-format parquet "
                f"--master 'local[*]'"
            ),
        )

    run_spark_etl
