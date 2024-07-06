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
import json


def create_user(token: str,
                address: str,
                user_id: str,
                user_name: str,
                passwort: str,
                is_admin: bool,
                verify_connection: bool = True) -> str:
    path = "/v1.0alpha/user"
    json_body = {
        "id": user_id,
        "name": user_name,
        "password": passwort,
        "is_admin": is_admin,
    }
    body_str = json.dumps(json_body)
    return hanami_request.send_post_request(token,
                                            address,
                                            path,
                                            body_str,
                                            verify=verify_connection)


def get_user(token: str,
             address: str,
             user_id: str,
             verify_connection: bool = True) -> str:
    path = "/v1.0alpha/user"
    values = f'id={user_id}'
    return hanami_request.send_get_request(token,
                                           address,
                                           path,
                                           values,
                                           verify=verify_connection)


def list_users(token: str,
               address: str,
               verify_connection: bool = True) -> str:
    path = "/v1.0alpha/user/all"
    return hanami_request.send_get_request(token,
                                           address,
                                           path,
                                           "",
                                           verify=verify_connection)


def delete_user(token: str,
                address: str,
                user_id: str,
                verify_connection: bool = True) -> str:
    path = "/v1.0alpha/user"
    values = f'id={user_id}'
    return hanami_request.send_delete_request(token,
                                              address,
                                              path,
                                              values,
                                              verify=verify_connection)


def add_roject_to_user(token: str,
                       address: str,
                       user_id: str,
                       project_id: str,
                       role: str,
                       is_project_admin: bool,
                       verify_connection: bool = True) -> str:
    path = "/v1.0alpha/user/project"
    json_body = {
        "id": user_id,
        "project_id": project_id,
        "role": role,
        "is_project_admin": is_project_admin,
    }
    body_str = json.dumps(json_body)
    return hanami_request.send_post_request(token,
                                            address,
                                            path,
                                            body_str,
                                            verify=verify_connection)


def remove_project_fromUser(token: str,
                            address: str,
                            user_id: str,
                            project_id: str,
                            verify_connection: bool = True) -> str:
    path = "/v1.0alpha/user/project"
    values = f'project_id={project_id}&id={user_id}'
    return hanami_request.send_delete_request(token,
                                              address,
                                              path,
                                              values,
                                              verify=verify_connection)


def list_projects_of_user(token: str,
                          address: str,
                          verify_connection: bool = True) -> str:
    path = "/v1.0alpha/user/project"
    return hanami_request.send_get_request(token,
                                           address,
                                           path,
                                           "",
                                           verify=verify_connection)


def switch_project(token: str,
                   address: str,
                   project_id: str,
                   verify_connection: bool = True) -> str:
    path = "/v1.0alpha/user/project"
    json_body = {
        "project_id": project_id,
    }
    body_str = json.dumps(json_body)
    return hanami_request.send_post_request(token,
                                            address,
                                            path,
                                            body_str,
                                            verify=verify_connection)
