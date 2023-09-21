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

func UploadTrainData(name string, dataType string, data string) (bool, string) {
    path := "control/train_data"
    vars := ""
    jsonBody := fmt.Sprintf("{\"name\":%s,\"type\":%s,\"data\":%s}", name, dataType, data)
    return SendPost(path, vars, jsonBody)
}

func GetTrainData(uuid string, withData bool) (bool, string) {
    path := "control/train_data"
    vars := fmt.Sprintf("uuid=%s", uuid)
    if withData {
        vars += "&with_data=true"
    } else {
        vars += "&with_data=false"
    }
    return SendGet(path, vars)
}

func ListTrainData() (bool, string) {
    path := fmt.Sprintf("control/train_datas")
    vars := ""
    return SendGet(path, vars)
}
