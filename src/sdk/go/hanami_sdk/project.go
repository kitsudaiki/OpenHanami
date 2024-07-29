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

func CreateProject(address string, token string, projectId string, projectName string, skipTlsVerification bool) (map[string]interface{}, error) {
    path := "v1.0alpha/project"
    jsonBody := map[string]interface{}{
        "id": projectId,
        "name": projectName,
    }
    return SendPost(address, token, path, jsonBody, skipTlsVerification)
}

func GetProject(address string, token string, projectId string, skipTlsVerification bool) (map[string]interface{}, error) {
    path := "v1.0alpha/project"
    vars := map[string]string{ "id": projectId }
    return SendGet(address, token, path, vars, skipTlsVerification)
}

func ListProject(address string, token string, skipTlsVerification bool) (map[string]interface{}, error) {
    path := fmt.Sprintf("v1.0alpha/project/all")
    vars := map[string]string{}
    return SendGet(address, token, path, vars, skipTlsVerification)
}

func DeleteProject(address string, token string, projectId string, skipTlsVerification bool) (map[string]interface{}, error) {
    path := "v1.0alpha/project"
    vars := map[string]string{ "id": projectId }
    return SendDelete(address, token, path, vars, skipTlsVerification)
}
