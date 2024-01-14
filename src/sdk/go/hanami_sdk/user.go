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

func CreateUser(address string, token string, userId string, userName string, pw string, is_admin bool) (bool, string) {
    path := "control/v1/user"
    vars := ""
    jsonBody := fmt.Sprintf("{\"id\":\"%s\",\"name\":\"%s\",\"password\":\"%s\",\"is_admin\":%v}", 
                            userId, userName, pw, is_admin)
    return SendPost(address, token, path, vars, jsonBody)
}

func GetUser(address string, token string, userId string) (bool, string) {
    path := "control/v1/user"
    vars := fmt.Sprintf("id=%s", userId)
    return SendGet(address, token, path, vars)
}

func ListUser(address string, token string) (bool, string) {
    path := fmt.Sprintf("control/v1/user/all")
    vars := ""
    return SendGet(address, token, path, vars)
}

func DeleteUser(address string, token string, userId string) (bool, string) {
    path := "control/v1/user"
    vars := fmt.Sprintf("id=%s", userId)
    return SendDelete(address, token, path, vars)
}
