from airflow import DAG
from datetime import datetime

from airflow.providers.standard.operators.python import PythonOperator


def hello_world():
  print("Hello, World!")
with DAG('hello_world', start_date=datetime(2026, 1, 1), schedule='@daily') as dag:
  task = PythonOperator(task_id='hello_task', python_callable=hello_world)
