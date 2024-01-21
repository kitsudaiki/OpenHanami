#!python3

import numpy as np
import matplotlib.pyplot as plt
from hanami_sdk import hanami_token
from hanami_sdk import checkpoint
from hanami_sdk import cluster
from hanami_sdk import dataset
from hanami_sdk import direct_io
from hanami_sdk import logs
from hanami_sdk import hosts
from hanami_sdk import project
from hanami_sdk import request_result
from hanami_sdk import task
from hanami_sdk import user
from hanami_sdk import hanami_exceptions
import test_values
import json
import time
import base64
import configparser


config = configparser.ConfigParser()
config.read('/etc/hanami/hanami_testing.conf')

address = config["connection"]["address"]
test_user_id = config["connection"]["test_user"]
test_user_pw = config["connection"]["test_pw"]

train_inputs = "/home/neptune/Schreibtisch/Projekte/Hanami/testing/python_sdk_api/train.csv"
request_inputs = "/home/neptune/Schreibtisch/Projekte/Hanami/testing/python_sdk_api/test.csv"

cluster_template = \
    "version: 1\n" \
    "settings:\n" \
    "   neuron_cooldown: 100000000000.5\n" \
    "   refractory_time: 1\n" \
    "   max_connection_distance: 1\n" \
    "   enable_reduction: false\n" \
    "bricks:\n" \
    "    1,1,1\n" \
    "        input: test_input\n" \
    "        number_of_neurons: 25\n" \
    "    2,1,1\n" \
    "        number_of_neurons: 128\n" \
    "    3,1,1\n" \
    "        number_of_neurons: 128\n" \
    "    4,1,1\n" \
    "        output: test_output\n" \
    "        number_of_neurons: 5"

cluster_name = "test_cluster"
generic_task_name = "test_task"
template_name = "dynamic"
request_dataset_name = "request_test_dataset"
train_dataset_name = "train_test_dataset"

token = hanami_token.request_token(address, test_user_id, test_user_pw)

result = cluster.create_cluster(token, address, cluster_name, cluster_template)
cluster_uuid = json.loads(result)["uuid"]

train_dataset_uuid = dataset.upload_csv_files(token, address, train_dataset_name, train_inputs)
request_dataset_uuid = dataset.upload_csv_files(token, address, request_dataset_name, request_inputs)





# train
for i in range(0,100):
    print("poi: ", i)
    result = task.create_task(token, address, generic_task_name, "train", cluster_uuid, train_dataset_uuid)
    task_uuid = json.loads(result)["uuid"]
    finished = False
    while not finished:
        result = task.get_task(token, address, task_uuid, cluster_uuid)
        finished = json.loads(result)["state"] == "finished"
        print("wait for finish train-task")
        time.sleep(0.2)
    result = task.delete_task(token, address, task_uuid, cluster_uuid)


# test
result = task.create_task(token, address, generic_task_name, "request", cluster_uuid, request_dataset_uuid)
task_uuid = json.loads(result)["uuid"]

finished = False
while not finished:
    result = task.get_task(token, address, task_uuid, cluster_uuid)
    finished = json.loads(result)["state"] == "finished"
    print("wait for finish request-task")
    time.sleep(0.2)
result = task.delete_task(token, address, task_uuid, cluster_uuid)



result = request_result.get_request_result(token, address, task_uuid)
data = json.loads(result)["data"]

dataset.delete_dataset(token, address, train_dataset_uuid)
dataset.delete_dataset(token, address, request_dataset_uuid)
cluster.delete_cluster(token, address, cluster_uuid)


# Open the file in write mode
with open("out.txt", 'w') as file:
    # Write each value from the array to a new line
    for value in data:
        file.write(str(value) + '\n')



plt.rcParams["figure.figsize"] = [10, 5]
plt.rcParams["figure.autolayout"] = True

plt.plot(data, color="red")

plt.show()
