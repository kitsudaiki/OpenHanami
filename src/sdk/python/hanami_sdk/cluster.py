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
from websockets.sync.client import connect
import json
import base64 


def create_cluster(token: str,
                   address: str, 
                   name: str,
                   template: str) -> str:
    # convert template to base64
    template_bytes = template.encode("ascii") 
    base64_bytes = base64.b64encode(template_bytes) 
    base64_string = base64_bytes.decode("ascii") 

    path = "/control/v1/cluster"
    json_body = {
        "name": name,
        "template": base64_string,
    }
    body_str = json.dumps(json_body)
    return hanami_request.send_post_request(token, address, path, body_str)


def save_cluster(token: str, address: str, name: str, cluster_uuid: str) -> str:
    path = "/control/v1/cluster/save"
    json_body = {
        "name": name,
        "cluster_uuid": cluster_uuid,
    }
    body_str = json.dumps(json_body)
    return hanami_request.send_post_request(token, address, path, body_str)


def restore_cluster(token: str, address: str, checkpoint_uuid: str, cluster_uuid: str) -> str:
    path = "/control/v1/cluster/load"
    json_body = {
        "checkpoint_uuid": checkpoint_uuid,
        "cluster_uuid": cluster_uuid,
    }
    body_str = json.dumps(json_body)
    return hanami_request.send_post_request(token, address, path, body_str)


def get_cluster(token: str, address: str, cluster_uuid: str) -> str:
    path = "/control/v1/cluster"
    values = f'uuid={cluster_uuid}'
    return hanami_request.send_get_request(token, address, path, values)


def list_clusters(token: str, address: str) -> str:
    path = "/control/v1/cluster/all"
    return hanami_request.send_get_request(token, address, path, "")


def delete_cluster(token: str, address: str, cluster_uuid: str) -> str:
    path = "/control/v1/cluster"
    values = f'uuid={cluster_uuid}'
    return hanami_request.send_delete_request(token, address, path, values)


def switch_to_task_mode(token: str, address: str, cluster_uuid: str):
    path = "/control/v1/cluster/set_mode"
    json_body = {
        "new_state": "TASK",
        "uuid": cluster_uuid,
    }
    body_str = json.dumps(json_body)
    return hanami_request.send_put_request(token, address, path, body_str)


def switch_to_direct_mode(token: str, address: str, cluster_uuid: str):
    # create initial request for the websocket-connection
    initial_ws_msg = {
        "token": token,
        "target": "cluster",
        "uuid": cluster_uuid,
    }
    body_str = json.dumps(initial_ws_msg)

    base_address = address.split('/')[2]
    ws = connect("ws://" + base_address)
    ws.send(body_str)
    message = ws.recv()
    result_json = json.loads(message)

    if result_json["success"] is False:
        return

    return ws
