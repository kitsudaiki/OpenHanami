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

func CreateCluster(address string, token string, name string, template string, skipTlsVerification bool) (map[string]interface{}, error) {
    path := "v1.0alpha/cluster"
    jsonBody := map[string]interface{}{ 
        "name": name,
        "template": b64.StdEncoding.EncodeToString([]byte(template)),
    }
    return SendPost(address, token, path, jsonBody, skipTlsVerification)
}

func GetCluster(address string, token string, clusterUuid string, skipTlsVerification bool) (map[string]interface{}, error) {
    path := "v1.0alpha/cluster"
    vars := map[string]string{ "uuid": clusterUuid }
    return SendGet(address, token, path, vars, skipTlsVerification)
}

func ListCluster(address string, token string, skipTlsVerification bool) (map[string]interface{}, error) {
    path := fmt.Sprintf("v1.0alpha/cluster/all")
    vars := map[string]string{}
    return SendGet(address, token, path, vars, skipTlsVerification)
}

func DeleteCluster(address string, token string, clusterUuid string, skipTlsVerification bool) (map[string]interface{}, error) {
    path := "v1.0alpha/cluster"
    vars := map[string]string{ "uuid": clusterUuid }
    return SendDelete(address, token, path, vars, skipTlsVerification)
}
 
func SaveCluster(address string, token string, clusterUuid string, checkpointName string, skipTlsVerification bool) (map[string]interface{}, error) {
    path := "v1.0alpha/cluster/save"
    jsonBody := map[string]interface{}{ 
        "name": checkpointName,
        "cluster_uuid": clusterUuid,
    }
    return SendPost(address, token, path, jsonBody, skipTlsVerification)
}

func RestoreCluster(address string, token string, clusterUuid string, checkpointUuid string, skipTlsVerification bool) (map[string]interface{}, error) {
    path := "v1.0alpha/cluster/load"
    jsonBody := map[string]interface{}{ 
        "checkpoint_uuid": checkpointUuid,
        "cluster_uuid": clusterUuid,
    }
    return SendPost(address, token, path, jsonBody, skipTlsVerification)
}
