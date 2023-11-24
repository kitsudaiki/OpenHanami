#!python3

# Copyright 2022 Tobias Anker
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from hanami_sdk import token as hanami_token
from hanami_sdk import checkpoint
from hanami_sdk import cluster
from hanami_sdk import dataset
from hanami_sdk import direct_io
from hanami_sdk import logs
from hanami_sdk import project
from hanami_sdk import request_result
from hanami_sdk import task
from hanami_sdk import user
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

train_inputs = config["test_data"]["train_inputs"]
train_labels = config["test_data"]["train_labels"]
request_inputs = config["test_data"]["request_inputs"]
request_labels = config["test_data"]["request_labels"]

cluster_template = \
    "version: 1\n" \
    "settings:\n" \
    "    max_synapse_sections: 1000\n" \
    "    sign_neg: 0.5\n" \
    "        \n" \
    "bricks:\n" \
    "    1,1,1\n" \
    "        input: test_input\n" \
    "        number_of_neurons: 784\n" \
    "    2,1,1\n" \
    "        number_of_neurons: 400\n" \
    "    3,1,1\n" \
    "        output: test_output\n" \
    "        number_of_neurons: 10"

user_id = "tsugumi"
user_name = "Tsugumi"
password = "new password"
is_admin = True
role = "tester"
projet_id = "test_project"
project_name = "Test Project"

cluster_name = "test_cluster"
checkpoint_name = "test_checkpoint"
generic_task_name = "test_task"
template_name = "dynamic"
request_dataset_name = "request_test_dataset"
train_dataset_name = "train_test_dataset"


def delete_all_cluster():
    success, result = cluster.list_clusters(token, address)
    body = json.loads(result)["body"]

    for entry in body:
        cluster.delete_cluster(token, address, entry[0])


def delete_all_projects():
    success, result = project.list_projects(token, address)
    body = json.loads(result)["body"]

    for entry in body:
        project.delete_project(token, address, entry[0])


def delete_all_user():
    success, result = user.list_users(token, address)
    body = json.loads(result)["body"]

    for entry in body:
        user.delete_user(token, address, entry[0])


def delete_all_datasets():
    success, result = dataset.list_datasets(token, address)
    print(result)

    body = json.loads(result)["body"]

    for entry in body:
        dataset.delete_dataset(token, address, entry[0])


def test_project():
    print("test project")

    success, result = project.create_project(token, address, projet_id, project_name)
    assert success
    success, result = project.create_project(token, address, projet_id, project_name)
    assert not success
    success, result = project.list_projects(token, address)
    assert success
    success, result = project.get_project(token, address, projet_id)
    assert success
    success, result = project.get_project(token, address, "fail_project")
    assert not success
    success, result = project.delete_project(token, address, projet_id)
    assert success
    success, result = project.delete_project(token, address, "fail_project")
    assert not success


def test_user():
    print("test user")

    success, result = user.create_user(token, address, user_id, user_name, password, is_admin)
    assert success
    success, result = user.create_user(token, address, user_id, user_name, password, is_admin)
    assert not success
    success, result = user.list_users(token, address)
    assert success
    success, result = user.get_user(token, address, user_id)
    assert success
    success, result = user.get_user(token, address, "fail_user")
    assert not success
    success, result = user.delete_user(token, address, user_id)
    assert success
    success, result = user.delete_user(token, address, "fail_user")
    assert not success


def test_dataset():
    print("test dataset")

    success, result = dataset.upload_mnist_files(
        token, address, train_dataset_name, train_inputs, train_labels)
    assert success
    dataset_uuid = result
    success, result = dataset.list_datasets(token, address)
    assert success
    success, result = dataset.get_dataset(token, address, dataset_uuid)
    assert success
    success, result = dataset.get_dataset(token, address, "fail_dataset")
    assert not success
    success, result = dataset.delete_dataset(token, address, dataset_uuid)
    assert success
    success, result = dataset.delete_dataset(token, address, "fail_dataset")
    assert not success


def test_cluster():
    print("test cluster")

    success, result = cluster.create_cluster(token, address, cluster_name, cluster_template)
    assert success
    cluster_uuid = json.loads(result)["uuid"]
    success, result = cluster.list_clusters(token, address)
    assert success
    success, result = cluster.get_cluster(token, address, cluster_uuid)
    assert success
    success, result = cluster.get_cluster(token, address, "fail_cluster")
    assert not success
    success, result = cluster.delete_cluster(token, address, cluster_uuid)
    assert success
    success, result = cluster.delete_cluster(token, address, "fail_cluster")
    assert not success


def test_workflow():
    print("test workflow")

    # init
    success, result = cluster.create_cluster(token, address, cluster_name, cluster_template)
    cluster_uuid = json.loads(result)["uuid"]
    success, train_dataset_uuid = dataset.upload_mnist_files(
        token, address, train_dataset_name, train_inputs, train_labels)
    success, request_dataset_uuid = dataset.upload_mnist_files(
        token, address, request_dataset_name, request_inputs, request_labels)

    # run training
    success, result = task.create_task(
        token, address, generic_task_name, "train", cluster_uuid, train_dataset_uuid)
    assert success
    task_uuid = json.loads(result)["uuid"]

    finished = False
    while not finished:
        success, result = task.get_task(token, address, task_uuid, cluster_uuid)
        assert success
        finished = json.loads(result)["state"] == "finished"
        print("wait for finish train-task")
        time.sleep(1)

    success, result = task.delete_task(token, address, task_uuid, cluster_uuid)
    assert success

    # save and reload checkpoint
    # success, result = cluster.save_cluster(token, address, checkpoint_name, cluster_uuid)
    # assert success
    # checkpoint_uuid = json.loads(result)["uuid"]

    # cluster.delete_cluster(token, address, cluster_uuid)
    # success, result = cluster.create_cluster(token, address, cluster_name, cluster_template)
    # cluster_uuid = json.loads(result)["uuid"]

    # success, result = cluster.restore_cluster(token, address, checkpoint_uuid, cluster_uuid)
    # assert success

    # run testing
    success, result = task.create_task(
        token, address, generic_task_name, "request", cluster_uuid, request_dataset_uuid)
    assert success
    task_uuid = json.loads(result)["uuid"]

    finished = False
    while not finished:
        success, result = task.get_task(token, address, task_uuid, cluster_uuid)
        assert success
        finished = json.loads(result)["state"] == "finished"
        print("wait for finish request-task")
        time.sleep(1)

    success, result = task.delete_task(token, address, task_uuid, cluster_uuid)
    assert success

    # check request-result
    success, result = request_result.get_request_result(token, address, task_uuid)
    assert success
    success, result = request_result.list_request_results(token, address)
    assert success
    success, result = request_result.check_against_dataset(
        token, address, task_uuid, request_dataset_uuid)
    assert success
    correctness = json.loads(result)["correctness"]
    print("=======================================")
    print("test-result: " + str(correctness))
    print("=======================================")
    assert correctness > 90.0
    success, result = request_result.delete_request_result(token, address, task_uuid)
    assert success

    # check direct-mode
    ws = cluster.switch_to_direct_mode(token, address, cluster_uuid)
    for i in range(0, 100):
        direct_io.send_train_input(ws, "test_input", test_values.get_direct_io_test_intput(), False)
        direct_io.send_train_input(ws, "test_output", test_values.get_direct_io_test_output(), True)
    output_values = direct_io.send_request_input(ws, "test_input", test_values.get_direct_io_test_intput(), True)
    # print(output_values)
    ws.close()

    assert list(output_values).index(max(output_values)) == 5

    # cleanup
    dataset.delete_dataset(token, address, train_dataset_uuid)
    dataset.delete_dataset(token, address, request_dataset_uuid)
    cluster.delete_cluster(token, address, cluster_uuid)


success, token = hanami_token.request_token(address, test_user_id, test_user_pw)
assert success

if success:

    delete_all_datasets()
    delete_all_cluster()
    delete_all_projects()
    delete_all_user()

    test_project()
    test_user()
    test_dataset()
    test_cluster()
    test_workflow()
