/**
 * @file        user_commands.go
  *
 * @author      Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright   Apache License Version 2.0
 *
 *      Copyright 2021 Tobias Anker
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

package http_request

import (
    "fmt"
)

func CreateUser_Request(id string, pw string, is_admin string, groups string) (bool, string) {
	path := "control/v1/create_user"
	vars := ""
	jsonBody := fmt.Sprintf("{\"id\":\"%s\",\"pw\":\"%s\",\"is_admin\":%s,\"groups\":\"%s\"}", 
	                        id, pw, is_admin, groups)
    return SendPost_Request(path, vars, jsonBody)
}

func GetUser_Request(id string) (bool, string) {
	path := "control/v1/user"
	vars := fmt.Sprintf("id=%s", id)
    return SendGet_Request(path, vars)
}

func ListUser_Request() (bool, string) {
	path := fmt.Sprintf("control/v1/user/all")
	vars := ""
    return SendGet_Request(path, vars)
}

func DeleteUser_Request(id string) (bool, string) {
	path := "control/v1/user"
	vars := fmt.Sprintf("id=%s", id)
    return SendDelete_Request(path, vars)
}