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
    "strconv"
)

func listAuditLogs(address string, token string, userId string, page int, skipTlsVerification bool) (map[string]interface{}, error) {
    path := "/v1.0alpha/audit_log?";
    vars := map[string]string{}
    if(userId != "") {
        vars = map[string]string{ 
            "user_id": userId,
            "page": strconv.Itoa(page),
        }
    } else {
        vars = map[string]string{ 
            "page": strconv.Itoa(page),
        }
    }
    return SendGet(address, token, path, vars, skipTlsVerification)
}

func listErrorLogs(address string, token string, userId string, page int, skipTlsVerification bool)  (map[string]interface{}, error) {
    path := "/v1.0alpha/error_log";
    vars := map[string]string{ 
        "user_id": userId,
        "page": strconv.Itoa(page),
    }
    return SendGet(address, token, path, vars, skipTlsVerification)
}
