from google.cloud import bigquery
from google.oauth2 import service_account
import os
from google.cloud import dataproc_v1
from google.cloud.dataproc_v1.gapic.transports import (job_controller_grpc_transport)
from google.cloud.dataproc_v1.gapic.transports import (
    cluster_controller_grpc_transport)
import webbrowser


def upload_to_bigquery(filepath,tablename):
    # TODO(developer): Set key_path to the path to the service account key
    #                  file.
    key_path = ''

    credentials = service_account.Credentials.from_service_account_file(
        key_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"], )

    client = bigquery.Client(
        credentials=credentials,
        project=credentials.project_id,)
    
    filename = filepath
    dataset_id = 'enter dataset id of bigQuery where the files has to be uploaded as tables'
    table_id = tablename

    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)
    job_config = bigquery.LoadJobConfig()
    job_config.source_format = bigquery.SourceFormat.CSV
    job_config.skip_leading_rows = 1
    job_config.autodetect = True

    with open(filename, "rb") as source_file:
        job = client.load_table_from_file(source_file, table_ref, job_config=job_config)

    job.result()  # Waits for table load to complete.

    print("Loaded {} rows into {}:{}.".format(job.output_rows, dataset_id, table_id))

    bucket_name = 'Enter name of your bucket'

    destination_uri = "gs://{}/bqdata/{}.csv".format(bucket_name,tablename)
    dataset_ref = client.dataset(dataset_id, project=credentials.project_id)
    table_ref = dataset_ref.table(table_id)

    extract_job = client.extract_table(
        table_ref,
        destination_uri,
        # Location must match that of the source table.
        location="US",
    )  # API request
    extract_job.result()  # Waits for job to complete.

    print(
        "Exported {}:{}.{} to {}".format(project, dataset_id, table_id, destination_uri)
    )

    
    return None

def submit_pyspark_job(dataproc, project, region, cluster_name, bucket_name,
                       filename,table_id):
    """Submit the Pyspark job to the cluster (assumes `filename` was uploaded
    to `bucket_name."""
    job_details = {
        'placement': {
            'cluster_name': cluster_name
        },
        'pyspark_job': {
            'main_python_file_uri': 'gs://{}/submit_jobs/{}'.format(bucket_name, filename),
            "args": ["gs://{}/bqdata/{}.csv".format(bucket_name,table_id),table_id]
        }
    }

    result = dataproc.submit_job(
        project_id=project, region=region, job=job_details)
    job_id = result.reference.job_id
    print('Submitted job ID {}.'.format(job_id))
    return job_id

def wait_for_job(dataproc, project, region, job_id):
    """Wait for job to complete or error out."""
    print('Waiting for job to finish...')
    while True:
        job = dataproc.get_job(project, region, job_id)
        # Handle exceptions
        if job.status.State.Name(job.status.state) == 'ERROR':
            raise Exception(job.status.details)
        elif job.status.State.Name(job.status.state) == 'DONE':
            print('Job finished.')
            return job


project = 'enter project id'
region = 'enter region of cluster'
cluster_name = 'enter cluster name'
bucket_name = 'enter bucket name'

job_transport = (job_controller_grpc_transport.JobControllerGrpcTransport(
                address='{}-dataproc.googleapis.com:443'.format(region)))
dataproc_job_client = dataproc_v1.JobControllerClient(job_transport)



##GUI Code

import PySimpleGUI as sg

sg.change_look_and_feel('Light Blue 2')
layout = [[sg.Text('Select Offline to train and Online to Predict')],
          [sg.Text('Mode', size=(15, 1)),sg.Drop(values=('Offline', 'Online'), auto_size_text=True)], 
          [sg.Text('Enter data path')],
          [sg.Text('File Path:', size=(8, 1)), sg.Input(), sg.FileBrowse()],
          [sg.Text('Enter data table name')],
          [sg.Text('TableName:', size=(8, 1)), sg.Input()],
          [sg.Button('Submit'), sg.Button('Exit')]]





window = sg.Window('AdTracking Fraud Detection', layout)

while True:
    event, values = window.read()
    if event == 'Submit':

        mode = values[0]
        if(mode == 'Offline'):
            train_job_file = 'Save_Final_Model.py'
            trainfile_path = values[1]
            traintable_name = values[2]
            upload_to_bigquery(trainfile_path,traintable_name)
            train_job_id = submit_pyspark_job(dataproc_job_client,project, region, cluster_name, bucket_name, train_job_file, traintable_name)
            wait_for_job(dataproc_job_client, project, region, train_job_id)
            sg.Popup('Model has been updated with new training data!')

        if(mode == 'Online'):
            predict_job_file = 'Predict_on_test_and_save_to_big_query.py'
            testfile_path = values[1]
            testtable_name = values[2]
            upload_to_bigquery(testfile_path,testtable_name)
            test_job_id = submit_pyspark_job(dataproc_job_client,project, region, cluster_name, bucket_name, predict_job_file, testtable_name)
            wait_for_job(dataproc_job_client, project, region, test_job_id)
            layout_online = [[sg.Text('Predictions are Ready. Look at IPs to Block!')],
            [sg.Button('Visualize'), sg.Button('Close')]]
            window_online = sg.Window('Visualize', layout_online)
            event_online, values_online = window_online.read()
            if event_online == 'Visualize':
                webbrowser.open('https://datastudio.google.com/s/sYGuCqCl87k')
            window_online.close()

    if event in (None, 'Exit'):      
        break

window.close()
