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
    b64 "encoding/base64"

    "fmt"
)

func CreateCluster(name string, template string) (bool, string) {
    sEnc := b64.StdEncoding.EncodeToString([]byte(template))
	jsonBody := fmt.Sprintf("{\"name\":\"%s\", \"template\":\"%s\"}", name, sEnc)
	path := "control/v1/cluster"
	vars := ""
    return SendPost(path, vars, jsonBody)
}

func GetCluster(clusterUuid string) (bool, string) {
    path := "control/v1/cluster"
    vars := fmt.Sprintf("uuid=%s", clusterUuid)
    return SendGet(path, vars)
}

func ListCluster() (bool, string) {
    path := fmt.Sprintf("control/v1/cluster/all")
    vars := ""
    return SendGet(path, vars)
}

func DeleteCluster(clusterUuid string) (bool, string) {
    path := "control/v1/cluster"
    vars := fmt.Sprintf("uuid=%s", clusterUuid)
    return SendDelete(path, vars)
}
 
func SaveCluster(clusterUuid string, checkpointName string) (bool, string) {
	jsonBody := fmt.Sprintf("{\"name\":\"%s\", \"cluster_uuid\":\"%s\"}", checkpointName, clusterUuid)
	path := "/control/v1/cluster/save"
	vars := ""
    return SendPost(path, vars, jsonBody)
}

func RestoreCluster(clusterUuid string, checkpointUuid string) (bool, string) {
	jsonBody := fmt.Sprintf("{\"checkpoint_uuid\":\"%s\", \"cluster_uuid\":\"%s\"}", checkpointUuid, clusterUuid)
	path := "/control/v1/cluster/load"
	vars := ""
    return SendPost(path, vars, jsonBody)
}
