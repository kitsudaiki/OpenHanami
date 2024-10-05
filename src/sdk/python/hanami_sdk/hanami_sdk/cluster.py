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
import websockets
import json
import base64
import ssl


def create_cluster(token: str,
                   address: str,
                   name: str,
                   template: str,
                   verify_connection: bool = True) -> str:
    # convert template to base64
    template_bytes = template.encode("ascii")
    base64_bytes = base64.b64encode(template_bytes)
    base64_string = base64_bytes.decode("ascii")

    path = "/v1.0alpha/cluster"
    json_body = {
        "name": name,
        "template": base64_string,
    }
    body_str = json.dumps(json_body)
    return hanami_request.send_post_request(token,
                                            address,
                                            path,
                                            body_str,
                                            verify=verify_connection)


def save_cluster(token: str,
                 address: str,
                 name: str,
                 cluster_uuid: str,
                 verify_connection: bool = True) -> str:
    path = "/v1.0alpha/cluster/save"
    json_body = {
        "name": name,
        "cluster_uuid": cluster_uuid,
    }
    body_str = json.dumps(json_body)
    return hanami_request.send_post_request(token,
                                            address,
                                            path,
                                            body_str,
                                            verify=verify_connection)


def restore_cluster(token: str,
                    address: str,
                    checkpoint_uuid: str,
                    cluster_uuid: str,
                    verify_connection: bool = True) -> str:
    path = "/v1.0alpha/cluster/load"
    json_body = {
        "checkpoint_uuid": checkpoint_uuid,
        "cluster_uuid": cluster_uuid,
    }
    body_str = json.dumps(json_body)
    return hanami_request.send_post_request(token,
                                            address,
                                            path,
                                            body_str,
                                            verify=verify_connection)


def get_cluster(token: str,
                address: str,
                cluster_uuid: str,
                verify_connection: bool = True) -> str:
    path = "/v1.0alpha/cluster"
    values = f'uuid={cluster_uuid}'
    return hanami_request.send_get_request(token,
                                           address,
                                           path,
                                           values,
                                           verify=verify_connection)


def list_clusters(token: str,
                  address: str,
                  verify_connection: bool = True) -> str:
    path = "/v1.0alpha/cluster/all"
    return hanami_request.send_get_request(token,
                                           address,
                                           path,
                                           "",
                                           verify=verify_connection)


def delete_cluster(token: str,
                   address: str,
                   cluster_uuid: str,
                   verify_connection: bool = True) -> str:
    path = "/v1.0alpha/cluster"
    values = f'uuid={cluster_uuid}'
    return hanami_request.send_delete_request(token,
                                              address,
                                              path,
                                              values,
                                              verify=verify_connection)


def switch_to_task_mode(token: str,
                        address: str,
                        cluster_uuid: str,
                        verify_connection: bool = True):
    path = "/v1.0alpha/cluster/set_mode"
    json_body = {
        "new_state": "TASK",
        "uuid": cluster_uuid,
    }
    body_str = json.dumps(json_body)
    return hanami_request.send_put_request(token,
                                           address,
                                           path,
                                           body_str,
                                           verify=verify_connection)


def switch_host(token: str,
                address: str,
                cluster_uuid: str,
                host_uuid: str,
                verify_connection: bool = True):
    path = "/v1.0alpha/cluster/switch_host"
    json_body = {
        "cluster_uuid": cluster_uuid,
        "host_uuid": host_uuid,
        "hexagon_id": 1,
    }
    body_str = json.dumps(json_body)
    return hanami_request.send_put_request(token,
                                           address,
                                           path,
                                           body_str,
                                           verify=verify_connection)


async def switch_to_direct_mode(token: str,
                                address: str,
                                cluster_uuid: str,
                                verify_connection: bool = True):
    path = "/v1.0alpha/cluster/set_mode"
    json_body = {
        "new_state": "DIRECT",
        "uuid": cluster_uuid,
    }
    body_str = json.dumps(json_body)
    hanami_request.send_put_request(token,
                                    address,
                                    path,
                                    body_str,
                                    verify=verify_connection)

    # create initial request for the websocket-connection
    initial_ws_msg = {
        "token": token,
        "target": "cluster",
        "uuid": cluster_uuid,
    }
    body_str = json.dumps(initial_ws_msg)

    ssl_context = None
    websocket_begin = "ws"
    if address.startswith("https"):
        websocket_begin = "wss"

        # Disable SSL verification
        if not verify_connection:
            ssl_context = ssl.SSLContext()
            ssl_context.verify_mode = ssl.CERT_NONE

    base_address = address.split('/')[2]
    ws = await websockets.connect(websocket_begin + "://" + base_address, ssl=ssl_context)

    await ws.send(body_str)
    message = await ws.recv()
    result_json = json.loads(message)

    if result_json["success"] is False:
        return

    return ws
