/**
 * @author      Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright   Apache License Version 2.0
 *
 *      Copyright 2022 Tobias Anker
 *
 *      Licensed under the Apache License, Version 2.0 (the "License");
 *      you may not use this file except in compliance with the License.
 *      You may obtain a copy of the License at
 *
 *          http://www.apache.org/licenses/LICENSE-2.0
 *
 *      Unless required by applicable law or agreed to in writing, software
 *      distributed under the License is distributed on an "AS IS" BASIS,
 *      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *      See the License for the specific language governing permissions and
 *      limitations under the License.
 */

package hanami_sdk

import (
    "fmt"
)

func CreateUser(address string, token string, userId string, userName string, pw string, is_admin bool) (map[string]interface{}, error) {
    path := "v1.0alpha/user"
    jsonBody := map[string]interface{}{ 
        "id": userId,
        "name": userName,
        "password": pw,
        "is_admin": is_admin,
    }
    return SendPost(address, token, path, jsonBody)
}

func GetUser(address string, token string, userId string) (map[string]interface{}, error) {
    path := "v1.0alpha/user"
    vars := map[string]string{ "id": userId }
    return SendGet(address, token, path, vars)
}

func ListUser(address string, token string) (map[string]interface{}, error) {
    path := fmt.Sprintf("v1.0alpha/user/all")
    vars := map[string]string{}
    return SendGet(address, token, path, vars)
}

func DeleteUser(address string, token string, userId string) (map[string]interface{}, error) {
    path := "v1.0alpha/user"
    vars := map[string]string{ "id": userId }
    return SendDelete(address, token, path, vars)
}
