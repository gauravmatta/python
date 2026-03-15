from airflow import DAG
from datetime import datetime
from airflow.providers.standard.operators.bash import BashOperator

default_args = {
'start_date': datetime(2022, 1, 1),
'schedule_interval': '@daily'
}
with DAG('my_dag', default_args=default_args) as dag:
    task1 = BashOperator(task_id='task1', bash_command='echo "Task 1"')
    task2 = BashOperator(task_id='task2', bash_command='echo "Task 2"')
    task3 = BashOperator(task_id='task3', bash_command='echo "Task 3"')
    task1 >> task2
    task1 >> task3