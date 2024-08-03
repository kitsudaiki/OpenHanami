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
	"strings"
)

func convertIO(data []string) map[string]string {
	valueMap := make(map[string]string)

	for _, val := range data {
		parts := strings.Split(val, ":")
		if len(parts) != 2 {
			break
		}
		valueMap[parts[0]] = parts[1]
	}

	return valueMap
}

func CreateTrainTask(address, token, name, clusterUuid string, inputs, outputs []string, skipTlsVerification bool) (map[string]interface{}, error) {
	path := "v1.0alpha/task/train"
	jsonBody := map[string]interface{}{
		"name":         name,
		"cluster_uuid": clusterUuid,
		"inputs":       convertIO(inputs),
		"outputs":      convertIO(outputs),
	}
	return SendPost(address, token, path, jsonBody, skipTlsVerification)
}

func CreateRequestTask(address, token, name, clusterUuid string, inputs, results []string, skipTlsVerification bool) (map[string]interface{}, error) {
	path := "v1.0alpha/task/request"
	jsonBody := map[string]interface{}{
		"name":         name,
		"cluster_uuid": clusterUuid,
		"inputs":       convertIO(inputs),
		"results":      convertIO(results),
	}
	return SendPost(address, token, path, jsonBody, skipTlsVerification)
}

func GetTask(address, token, taskId, clusterUuid string, skipTlsVerification bool) (map[string]interface{}, error) {
	path := "v1.0alpha/task"
	vars := map[string]interface{}{
		"uuid":         taskId,
		"cluster_uuid": clusterUuid,
	}
	return SendGet(address, token, path, vars, skipTlsVerification)
}

func ListTask(address, token, clusterUuid string, skipTlsVerification bool) (map[string]interface{}, error) {
	path := "v1.0alpha/task/all"
	vars := map[string]interface{}{"cluster_uuid": clusterUuid}
	return SendGet(address, token, path, vars, skipTlsVerification)
}

func DeleteTask(address, token, taskUuid, clusterUuid string, skipTlsVerification bool) (map[string]interface{}, error) {
	path := "v1.0alpha/task"
	vars := map[string]interface{}{
		"uuid":         taskUuid,
		"cluster_uuid": clusterUuid,
	}
	return SendDelete(address, token, path, vars, skipTlsVerification)
}
