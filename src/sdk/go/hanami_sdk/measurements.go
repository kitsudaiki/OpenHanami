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

func getPowerData(address string, token string) (bool, string) {
    path := "/control/v1/power_consumption";
    vars := ""
    return SendGet(address, token, path, vars)
}

func getTemperatureData_request(address string, token string) (bool, string) {
    path := "/control/v1/temperature_production";
    vars := ""
    return SendGet(address, token, path, vars)
}

func getSpeedData_request(address string, token string) (bool, string) {
    path := "/control/v1/speed";
    vars := ""
    return SendGet(address, token, path, vars)
}
