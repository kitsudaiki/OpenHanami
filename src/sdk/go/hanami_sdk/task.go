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

func CreateTask(address string, token string, name string, taskType string, clusterUuid string, dataSetUuid string) (bool, string) {
	path := "control/v1.0alpha/task"
	vars := ""
	jsonBody := fmt.Sprintf("{\"name\":%s, \"type\":%s, \"cluster_uuid\":%s, \"cluster_uuid\":\"%s\"}", 
                            name, 
                            taskType, 
                            clusterUuid, 
                            dataSetUuid)
    return SendPost(address, token, path, vars, jsonBody)
}

func GetTask(address string, token string, taskId string, clusterUuid string) (bool, string) {
    path := "control/v1.0alpha/task"
    vars := fmt.Sprintf("uuid=%s&cluster_uuid=%s", taskId, clusterUuid)
    return SendGet(address, token, path, vars)
}

func ListTask(address string, token string, clusterUuid string) (bool, string) {
    path := "control/v1.0alpha/task/all"
    vars := fmt.Sprintf("cluster_uuid=%s", clusterUuid)
    return SendGet(address, token, path, vars)
}

func DeleteTask(address string, token string, taskId string, clusterUuid string) (bool, string) {
    path := "control/v1.0alpha/task"
    vars := fmt.Sprintf("uuid=%s&cluster_uuid=%s", taskId, clusterUuid)
    return SendDelete(address, token, path, vars)
}
