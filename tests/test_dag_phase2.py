"""Phase 2 Airflow DAG tests.

Skipped cleanly when Airflow is not installed (it lives in its own
environment, see requirements-airflow.txt), but they do run in an environment
that has it -- previously nothing imported the DAG at all, so a syntax or
wiring error in it would have gone unnoticed.
"""

import pytest

pytest.importorskip("airflow")


@pytest.fixture()
def dag():
    from dags.etl_scheduler import dag as etl_dag

    return etl_dag


def test_dag_is_importable_with_expected_id(dag):
    assert dag.dag_id == "smartshop_daily_etl"
    assert dag.catchup is False


def test_dag_has_no_cycles(dag):
    from airflow.utils.dag_cycle_tester import check_cycle

    check_cycle(dag)


def test_local_environment_wires_ingest_before_etl(dag):
    task_ids = set(dag.task_ids)

    # Default (non-databricks) environment: raw data is materialised first,
    # then the Spark job consumes it.
    assert task_ids == {
        "materialize_amazon_reviews_2023_local",
        "run_spark_etl_job_local",
    }
    ingest = dag.get_task("materialize_amazon_reviews_2023_local")
    assert "run_spark_etl_job_local" in ingest.downstream_task_ids


def test_tasks_carry_retry_policy(dag):
    for task in dag.tasks:
        assert task.retries == 1
