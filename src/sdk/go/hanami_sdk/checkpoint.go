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

func GetCheckpoint(address string, token string, checkpointUuid string) (bool, string) {
    path := "control/v1.0alpha/checkpoint"
    vars := fmt.Sprintf("uuid=%s", checkpointUuid)
    return SendGet(address, token, path, vars)
}

func ListCheckpoint(address string, token string) (bool, string) {
    path := fmt.Sprintf("control/v1.0alpha/checkpoint/all")
    vars := ""
    return SendGet(address, token, path, vars)
}

func DeleteCheckpoint(address string, token string, checkpointUuid string) (bool, string) {
    path := "control/v1.0alpha/checkpoint"
    vars := fmt.Sprintf("uuid=%s", checkpointUuid)
    return SendDelete(address, token, path, vars)
}
 