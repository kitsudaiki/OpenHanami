# Copyright 2022 Tobias Anker
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

from websockets.sync.client import connect
from hanami_sdk import hanami_request
from hanami_sdk import proto3_pb2
import json
import asyncio
import math
import time


def list_datasets(token: str, address: str) -> tuple[bool,str]:
    path = "/control/v1/data_set/all"
    return hanami_request.send_get_request(token, address, path, "")


def get_dataset(token: str, address: str, dataset_uuid: str) -> tuple[bool,str]:
    path = "/control/v1/data_set"
    values = f'uuid={dataset_uuid}'
    return hanami_request.send_get_request(token, address, path, values)


def delete_dataset(token: str,
                   address: str, 
                   checkpoint_uuid: str) -> tuple[bool,str]:
    path = "/control/v1/data_set";
    values = f'uuid={checkpoint_uuid}'
    return hanami_request.send_delete_request(token, address, path, values)


def wait_until_upload_complete(token: str, address: str, uuid: str) -> bool:
    while True:
        path = "/control/v1/data_set/progress"
        values = f'uuid={uuid}'

        success, result = hanami_request.send_get_request(token, address, path, values)

        if not success:
            return False

        result_json = json.loads(result)
        if result_json["complete"]:
            return True

        time.sleep(0.5)

    return True


def send_data(token: str, 
              address: str, 
              dataset_uuid: str, 
              file_uuid: str, 
              data) -> bool:
    # create initial request for the websocket-connection
    initial_ws_msg = {
        "token": token,
        "target": "shiori",
    }
    body_str = json.dumps(initial_ws_msg)

    base_address = address.split('/')[2]
    with connect("ws://" + base_address) as websocket:
        websocket.send(body_str)
        message = websocket.recv()
        result_json = json.loads(message)

        if result_json["success"] is False:
            return False

        total_size = len(data)
        chunk_size = 96 * 1024
        num_chunks = math.ceil(total_size / chunk_size)

        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total_size)
            chunk_data = data[start:end]

            file_upload_msg = proto3_pb2.FileUpload_Message()
            file_upload_msg.datasetUuid = dataset_uuid
            file_upload_msg.fileUuid = file_uuid
            file_upload_msg.type = 0
            file_upload_msg.position = start
            file_upload_msg.isLast = (end - start) < chunk_size
            file_upload_msg.data = chunk_data
            serialized_msg = file_upload_msg.SerializeToString()

            websocket.send(serialized_msg)

            # message = websocket.recv()

    return True


def upload_mnist_files(token: str, 
                       address: str, 
                       name: str, 
                       input_file_path: str, 
                       label_file_path: str) -> tuple[bool,str]:
    # read files
    with open(input_file_path, 'rb') as i_f:
        input_file_data = i_f.read()

    with open(label_file_path, 'rb') as l_f:
        label_file_data = l_f.read()

    # initialize
    path = "/control/v1/mnist/data_set"
    json_body = {
        "name": name,
        "input_data_size": len(input_file_data),
        "label_data_size": len(label_file_data),
    }
    body_str = json.dumps(json_body)
    success, result = hanami_request.send_post_request(token, address, path, body_str)

    if not success:
        return False, ""

    # process init-result
    result_json = json.loads(result)
    uuid = result_json["uuid"]
    input_file_uuid = result_json["uuid_input_file"]
    label_file_uuid = result_json["uuid_label_file"]

    # send data
    if not send_data(token, address, uuid, input_file_uuid, input_file_data):
        return False, ""

    if not send_data(token, address, uuid, label_file_uuid, label_file_data):
        return False, ""

    if not wait_until_upload_complete(token, address, uuid): 
        return False, ""

    # finalize
    path = "/control/v1/mnist/data_set"
    json_body = {
        "uuid": uuid,
        "uuid_input_file": input_file_uuid,
        "uuid_label_file": label_file_uuid,
    }
    body_str = json.dumps(json_body)
    success, result = hanami_request.send_put_request(token, address, path, body_str)
    result_json = json.loads(result)

    if not success:
        return False, ""

    return True, uuid


def upload_csv_files(token: str, 
                     address: str, 
                     name: str, 
                     input_file_path: str) -> tuple[bool,str]:
    # read files
    with open(input_file_path, 'rb') as i_f:
        input_file_data = i_f.read()

    # initialize
    path = "/control/v1/csv/data_set"
    json_body = {
        "name": name,
        "input_data_size": len(input_file_data),
    }
    body_str = json.dumps(json_body)
    success, result = hanami_request.send_post_request(token, address, path, body_str)

    if not success:
        return False, ""

    # process init-result
    result_json = json.loads(result)
    uuid = result_json["uuid"]
    input_file_uuid = result_json["uuid_input_file"]

    # send data
    if not send_data(token, address, uuid, input_file_uuid, input_file_data):
        return False, ""

    if not wait_until_upload_complete(token, address, uuid): 
        return False, ""

    # finalize
    path = "/control/v1/csv/data_set"
    json_body = {
        "uuid": uuid,
        "uuid_input_file": input_file_uuid,
    }
    body_str = json.dumps(json_body)
    success, result = hanami_request.send_put_request(token, address, path, body_str)
    result_json = json.loads(result)

    if not success:
        return False, ""

    return True, uuid
