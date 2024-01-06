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


def create_project(token: str,
                   address: str, 
                   project_id: str, 
                   project_name: str) -> str:
    path = "/control/v1/project"
    json_body = {
        "id": project_id,
        "name": project_name,
    }
    body_str = json.dumps(json_body)
    return hanami_request.send_post_request(token, address, path, body_str)


def get_project(token: str, address: str, project_id: str) -> str:
    path = "/control/v1/project"
    values = f'id={project_id}'
    return hanami_request.send_get_request(token, address, path, values)


def list_projects(token: str, address: str) -> str:
    path = "/control/v1/project/all"
    return hanami_request.send_get_request(token, address, path, "")


def delete_project(token: str, address: str, project_id: str) -> str:
    path = "/control/v1/project"
    values = f'id={project_id}'
    return hanami_request.send_delete_request(token, address, path, values)
