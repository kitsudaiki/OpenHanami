/**
 * @file        train_data_commands.go
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

func UploadTrainData_Request(name string, dataType string, data string) (bool, string) {
	path := "control/train_data"
	vars := ""
	jsonBody := fmt.Sprintf("{\"name\":%s,\"type\":%s,\"data\":%s}", name, dataType, data)
    return SendPost_Request(path, vars, jsonBody)
}

func GetTrainData_Request(data_uuid string, with_data bool) (bool, string) {
	path := "control/train_data"
	vars := fmt.Sprintf("uuid=%s", data_uuid)
	if with_data {
		vars += "&with_data=true"
	} else {
		vars += "&with_data=false"
	}
    return SendGet_Request(path, vars)
}

func ListTrainData_Request() (bool, string) {
	path := fmt.Sprintf("control/train_datas")
	vars := ""
    return SendGet_Request(path, vars)
}
