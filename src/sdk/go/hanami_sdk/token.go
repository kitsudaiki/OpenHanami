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
    "encoding/json"
)

func parseJson(input string) map[string]interface{} {
    // parse json and fill into map
    outputMap := map[string]interface{}{}
    err := json.Unmarshal([]byte(input), &outputMap)
    if err != nil {
        panic(err)
    }

    return outputMap
}

func RequestToken(address string, user string, pw string) string {
    path := fmt.Sprintf("control/v1/token")
    body := fmt.Sprintf("{\"id\":\"%s\",\"password\":\"%s\"}", user, pw)

    success, content := sendGenericRequest(address, "", "POST", path, body)
    if success == false {
        return ""
    }

    outputMap := parseJson(content)
    return  outputMap["token"].(string)
}
