#!python3

import matplotlib
import matplotlib.pyplot as plt
from hanami_sdk import hanami_token
from hanami_sdk import cluster
from hanami_sdk import dataset
from hanami_sdk import task
import json
import time
import configparser

matplotlib.use('Qt5Agg')

def delete_all_cluster():
    result = cluster.list_clusters(token, address, False)
    body = json.loads(result)["body"]

    for entry in body:
        cluster.delete_cluster(token, address, entry[1], False)


def delete_all_datasets():
    result = dataset.list_datasets(token, address, False)
    body = json.loads(result)["body"]

    for entry in body:
        dataset.delete_dataset(token, address, entry[1], False)


config = configparser.ConfigParser()
config.read('/etc/openhanami/hanami_testing.conf')

address = config["connection"]["address"]
test_user_id = config["connection"]["test_user"]
test_user_pw = config["connection"]["test_passphrase"]

train_inputs = "/home/neptune/Schreibtisch/Projekte/OpenHanami/testing/python_sdk_api/train.csv"
request_inputs = "/home/neptune/Schreibtisch/Projekte/OpenHanami/testing/python_sdk_api/test.csv"

cluster_template = \
    "version: 1\n" \
    "settings:\n" \
    "   neuron_cooldown: 1.1\n" \
    "   refractory_time: 20\n" \
    "   max_connection_distance: 1\n" \
    "   enable_reduction: false\n" \
    "hexagons:\n" \
    "    1,1,1\n" \
    "    2,1,1\n" \
    "    3,1,1\n" \
    "    \n" \
    "inputs:\n" \
    "    test_input:\n" \
    "        target: 1,1,1\n" \
    "        number_of_inputs: 20\n" \
    "\n" \
    "outputs:\n" \
    "    test_output:\n" \
    "        target: 3,1,1\n" \
    "        number_of_outputs: 1\n"

cluster_name = "test_cluster"
generic_task_name = "test_task"
template_name = "dynamic"
request_dataset_name = "request_test_dataset"
train_dataset_name = "train_test_dataset"

token = hanami_token.request_token(address, test_user_id, test_user_pw)



delete_all_datasets()
delete_all_cluster()


result = cluster.create_cluster(token, address, cluster_name, cluster_template)
cluster_uuid = json.loads(result)["uuid"]

train_dataset_uuid = dataset.upload_csv_files(token, address, train_dataset_name, train_inputs)
request_dataset_uuid = dataset.upload_csv_files(
    token, address, request_dataset_name, request_inputs)


inputs = {
    "test_input": train_dataset_uuid
}

outputs = {
    "test_output": train_dataset_uuid
}


# train
for i in range(0, 100):
    print("poi: ", i)
    result = task.create_train_task(token, address, generic_task_name, cluster_uuid, inputs, outputs, 20)
    task_uuid = json.loads(result)["uuid"]
    finished = False
    while not finished:
        result = task.get_task(token, address, task_uuid, cluster_uuid)
        finished = json.loads(result)["state"] == "finished"
        # print(result)
        print("wait for finish train-task")
        time.sleep(0.2)
    result = task.delete_task(token, address, task_uuid, cluster_uuid)


inputs = {
    "test_input": request_dataset_uuid
}

results = {
    "test_output": "test_output"
}


# test
result = task.create_request_task(token, address, generic_task_name, cluster_uuid, inputs, results, 20)
task_uuid = json.loads(result)["uuid"]

finished = False
while not finished:
    result = task.get_task(token, address, task_uuid, cluster_uuid)
    finished = json.loads(result)["state"] == "finished"
    print(result)
    print("wait for finish request-task")
    time.sleep(0.2)
# result = task.delete_task(token, address, task_uuid, cluster_uuid)

result = dataset.download_dataset_content(
        token, address, task_uuid, "test_output", 1700, 0, False)

data = json.loads(result)["data"]
flattened_list = [item for sublist in data for item in sublist]

# Output the flattened list
print(flattened_list)
# print(json.dumps(json.loads(result), indent=4))

# dataset.delete_dataset(token, address, train_dataset_uuid)
# dataset.delete_dataset(token, address, request_dataset_uuid)
# cluster.delete_cluster(token, address, cluster_uuid)


# # Open the file in write mode
with open("out.txt", 'w') as file:
    # Write each value from the array to a new line
    for value in flattened_list:
        file.write(str(value) + '\n')


plt.rcParams["figure.figsize"] = [10, 5]
plt.rcParams["figure.autolayout"] = True

plt.plot(flattened_list, color="red")

plt.show()
