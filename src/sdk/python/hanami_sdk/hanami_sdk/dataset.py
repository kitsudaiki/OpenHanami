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

from . import hanami_request
from . import hanami_exceptions
from .hanami_messages import proto3_pb2
import json
import math
import time
import asyncio
import ssl
import websockets


def list_datasets(token: str,
                  address: str,
                  verify_connection: bool = True) -> str:
    path = "/control/v1/dataset/all"
    return hanami_request.send_get_request(token,
                                           address,
                                           path,
                                           "",
                                           verify=verify_connection)


def get_dataset(token: str,
                address: str,
                dataset_uuid: str,
                verify_connection: bool = True) -> str:
    path = "/control/v1/dataset"
    values = f'uuid={dataset_uuid}'
    return hanami_request.send_get_request(token,
                                           address,
                                           path,
                                           values,
                                           verify=verify_connection)


def delete_dataset(token: str,
                   address: str,
                   checkpoint_uuid: str,
                   verify_connection: bool = True) -> str:
    path = "/control/v1/dataset"
    values = f'uuid={checkpoint_uuid}'
    return hanami_request.send_delete_request(token,
                                              address,
                                              path, 
                                              values, 
                                              verify=verify_connection)


def check_mnist_dataset(token: str,
                        address: str,
                        dataset_uuid: str,
                        result_dataset_uuid: str,
                        verify_connection: bool = True) -> str:
    path = "/control/v1/dataset/check"
    values = f'dataset_uuid={dataset_uuid}&result_uuid={result_dataset_uuid}'
    return hanami_request.send_get_request(token,
                                           address,
                                           path, 
                                           values, 
                                           verify=verify_connection)


def wait_until_upload_complete(token: str,
                               address: str,
                               uuid: str,
                               verify_connection: bool = True) -> bool:
    while True:
        path = "/control/v1/dataset/progress"
        values = f'uuid={uuid}'

        result = hanami_request.send_get_request(token,
                                                 address,
                                                 path,
                                                 values,
                                                 verify=verify_connection)

        result_json = json.loads(result)
        if result_json["complete"]:
            return True

        time.sleep(0.5)

    return True


async def send_data(token: str,
                    address: str,
                    dataset_uuid: str,
                    file_uuid: str,
                    data,
                    verify_connection: bool = True) -> bool:
    # create initial request for the websocket-connection
    initial_ws_msg = {
        "token": token,
        "target": "file_upload",
        "uuid": file_uuid,
    }
    body_str = json.dumps(initial_ws_msg)

    ssl_context = None
    ws_begin = "ws"
    if address.startswith("https"):
        ws_begin = "wss"

        # Disable SSL verification
        if not verify_connection:
            ssl_context = ssl.SSLContext()
            ssl_context.verify_mode = ssl.CERT_NONE

    base_address = address.split('/')[2]

    async with websockets.connect(ws_begin + "://" + base_address, ssl=ssl_context) as websocket:
        await websocket.send(body_str)
        message = await websocket.recv()
        result_json = json.loads(message)

        if result_json["success"] is False:
            return False

        total_size = len(data)
        chunk_size = 128 * 1024
        num_chunks = math.ceil(total_size / chunk_size)

        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total_size)
            chunk_data = data[start:end]

            file_upload_msg = proto3_pb2.FileUpload_Message()
            file_upload_msg.position = start
            file_upload_msg.data = chunk_data
            serialized_msg = file_upload_msg.SerializeToString()

            await websocket.send(serialized_msg)
            message = await websocket.recv()

    return True


def upload_mnist_files(token: str,
                       address: str,
                       name: str,
                       input_file_path: str,
                       label_file_path: str,
                       verify_connection: bool = True) -> str:
    # read files
    with open(input_file_path, 'rb') as i_f:
        input_file_data = i_f.read()

    with open(label_file_path, 'rb') as l_f:
        label_file_data = l_f.read()

    # initialize
    path = "/control/v1/dataset/upload/mnist"
    json_body = {
        "name": name,
        "input_data_size": len(input_file_data),
        "label_data_size": len(label_file_data),
    }
    body_str = json.dumps(json_body)
    result = hanami_request.send_post_request(
        token, address, path, body_str, verify=verify_connection)

    # process init-result
    result_json = json.loads(result)
    uuid = result_json["uuid"]
    input_file_uuid = result_json["uuid_input_file"]
    label_file_uuid = result_json["uuid_label_file"]

    # send data
    if not asyncio.run(send_data(token,
                                 address,
                                 uuid,
                                 input_file_uuid,
                                 input_file_data,
                                 verify_connection=verify_connection)):
        raise hanami_exceptions.InternalServerErrorException()

    if not asyncio.run(send_data(token,
                                 address,
                                 uuid,
                                 label_file_uuid,
                                 label_file_data,
                                 verify_connection=verify_connection)):
        raise hanami_exceptions.InternalServerErrorException()

    if not wait_until_upload_complete(token,
                                      address,
                                      uuid,
                                      verify_connection=verify_connection):
        raise hanami_exceptions.InternalServerErrorException()

    # finalize
    path = "/control/v1/dataset/upload/mnist"
    json_body = {
        "uuid": uuid,
        "uuid_input_file": input_file_uuid,
        "uuid_label_file": label_file_uuid,
    }
    body_str = json.dumps(json_body)
    result = hanami_request.send_put_request(
        token, address, path, body_str, verify=verify_connection)
    result_json = json.loads(result)

    return uuid


def upload_csv_files(token: str,
                     address: str,
                     name: str,
                     input_file_path: str,
                     verify_connection: bool = True) -> str:
    # read files
    with open(input_file_path, 'rb') as i_f:
        input_file_data = i_f.read()

    # initialize
    path = "/control/v1/dataset/upload/csv"
    json_body = {
        "name": name,
        "input_data_size": len(input_file_data),
    }
    body_str = json.dumps(json_body)
    result = hanami_request.send_post_request(token,
                                              address,
                                              path,
                                              body_str,
                                              verify=verify_connection)

    # process init-result
    result_json = json.loads(result)
    uuid = result_json["uuid"]
    input_file_uuid = result_json["uuid_input_file"]

    # send data
    if not asyncio.run(send_data(token,
                                 address,
                                 uuid,
                                 input_file_uuid,
                                 input_file_data,
                                 verify_connection=verify_connection)):
        raise hanami_exceptions.InternalServerErrorException()

    if not wait_until_upload_complete(token,
                                      address,
                                      uuid,
                                      verify_connection=verify_connection):
        raise hanami_exceptions.InternalServerErrorException()

    # finalize
    path = "/control/v1/dataset/upload/csv"
    json_body = {
        "uuid": uuid,
        "uuid_input_file": input_file_uuid,
    }
    body_str = json.dumps(json_body)
    result = hanami_request.send_put_request(
        token, address, path, body_str, verify=verify_connection)
    result_json = json.loads(result)

    return uuid
