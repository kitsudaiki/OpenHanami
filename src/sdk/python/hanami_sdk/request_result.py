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

from hanami_sdk import hanami_request
import json


def get_request_result(token: str, address: str, request_result_uuid: str) -> tuple[bool,str]:
    path = "/control/v1/request_result"
    values = f'uuid={request_result_uuid}'
    return hanami_request.send_get_request(token, address, path, values)


def list_request_results(token: str, address: str) -> tuple[bool,str]:
    path = "/control/v1/request_result/all"
    return hanami_request.send_get_request(token, address, path, "")


def delete_request_result(token: str, address: str, request_result_uuid: str) -> tuple[bool,str]:
    path = "/control/v1/request_result"
    values = f'uuid={request_result_uuid}'
    return hanami_request.send_delete_request(token, address, path, values)


def check_against_dataset(token: str, address: str, result_uuid: str, dataset_uuid: str) -> tuple[bool,str]:
    path = "/control/v1/dataset/check"
    json_body = {
        "result_uuid": result_uuid,
        "dataset_uuid": dataset_uuid,
    }
    body_str = json.dumps(json_body)
    return hanami_request.send_post_request(token, address, path, body_str)
