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
import json


def send_post_request(token: str,
                      address: str, 
                      path: str, 
                      body: str) -> tuple[bool,str]:
    url = f'{address}{path}'
    headers = {'content-type': 'application/json'}
    headers = {'X-Auth-Token': token}
    response = requests.post(url, data=body, headers=headers) 
    if response.status_code != 200:
        return False, response.content
    else:
        return True, response.content


def send_get_request(token: str,
                     address: str, 
                     path: str, 
                     values: str) -> tuple[bool,str]:
    if values:
        url = f'{address}{path}?{values}'
    else:
        url = f'{address}{path}'

    headers = {'X-Auth-Token': token}
    response = requests.get(url, headers=headers) 
    if response.status_code != 200:
        return False, response.content
    else:
        return True, response.content


def send_put_request(token: str,
                     address: str, 
                     path: str, 
                     body: str) -> tuple[bool,str]:
    url = f'{address}{path}'
    headers = {'content-type': 'application/json'}
    headers = {'X-Auth-Token': token}
    response = requests.put(url, data=body, headers=headers)   
    if response.status_code != 200:
        return False, response.content
    else:
        return True, response.content


def send_delete_request(token: str,
                        address: str, 
                        path: str, 
                        values: str) -> tuple[bool,str]:
    url = f'{address}{path}?{values}'
    headers = {'X-Auth-Token': token}
    response = requests.delete(url, headers=headers) 
    if response.status_code != 200:
        return False, response.content
    else:
        return True, ""


