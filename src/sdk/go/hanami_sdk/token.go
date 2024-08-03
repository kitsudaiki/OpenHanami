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

func RequestToken(address, user, pw string, skipTlsVerification bool) string {
	path := "v1.0alpha/token"
	jsonBody := map[string]interface{}{
		"id":       user,
		"password": pw,
	}

	content, err := sendGenericRequest(address, "", "POST", path, &jsonBody, skipTlsVerification)
	if err != nil {
		return ""
	}

	return content["token"].(string)
}
