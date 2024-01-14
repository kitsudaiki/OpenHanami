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

func CreateProject(address string, token string, projectId string, projectName string) (bool, string) {
    path := "control/v1/project"
    vars := ""
    jsonBody := fmt.Sprintf("{\"id\":\"%s\",\"name\":\"%s\"}", 
                            projectId, projectName)
    return SendPost(address, token, path, vars, jsonBody)
}

func GetProject(address string, token string, projectId string) (bool, string) {
    path := "control/v1/project"
    vars := fmt.Sprintf("id=%s", projectId)
    return SendGet(address, token, path, vars)
}

func ListProject(address string, token string) (bool, string) {
    path := fmt.Sprintf("control/v1/project/all")
    vars := ""
    return SendGet(address, token, path, vars)
}

func DeleteProject(address string, token string, projectId string) (bool, string) {
    path := "control/v1/project"
    vars := fmt.Sprintf("id=%s", projectId)
    return SendDelete(address, token, path, vars)
}
