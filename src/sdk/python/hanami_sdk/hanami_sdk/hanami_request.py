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

import requests
from . import hanami_exceptions


def _handle_response(response) -> str:
    if response.status_code == 200:
        return response.content
    if response.status_code == 400:
        raise hanami_exceptions.BadRequestException(response.content)
    if response.status_code == 401:
        raise hanami_exceptions.UnauthorizedException(response.content)
    if response.status_code == 404:
        raise hanami_exceptions.NotFoundException(response.content)
    if response.status_code == 409:
        raise hanami_exceptions.ConflictException(response.content)
    if response.status_code == 500:
        raise hanami_exceptions.InternalServerErrorException()


def send_post_request(token: str,
                      address: str,
                      path: str,
                      body: str) -> str:
    url = f'{address}{path}'
    headers = {'content-type': 'application/json'}
    headers = {'X-Auth-Token': token}
    response = requests.post(url, data=body, headers=headers, verify=False)
    return _handle_response(response)


def send_get_request(token: str,
                     address: str,
                     path: str,
                     values: str) -> str:
    if values:
        url = f'{address}{path}?{values}'
    else:
        url = f'{address}{path}'

    headers = {'X-Auth-Token': token}
    response = requests.get(url, headers=headers, verify=False)
    return _handle_response(response)


def send_put_request(token: str,
                     address: str,
                     path: str,
                     body: str) -> str:
    url = f'{address}{path}'
    headers = {'content-type': 'application/json'}
    headers = {'X-Auth-Token': token}
    response = requests.put(url, data=body, headers=headers, verify=False)
    return _handle_response(response)


def send_delete_request(token: str,
                        address: str,
                        path: str,
                        values: str) -> str:
    url = f'{address}{path}?{values}'
    headers = {'X-Auth-Token': token}
    response = requests.delete(url, headers=headers, verify=False)
    return _handle_response(response)
