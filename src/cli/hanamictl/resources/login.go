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

package hanami_resources

import (
    "os"
    "hanamictl/common"
    "github.com/kitsudaiki/Hanami"
)

func Login() string {

    user := os.Getenv("HANAMI_USER")
    pw := os.Getenv("HANAMI_PW")
    address := os.Getenv("HANAMI_ADDRESS")

    if user == "" {
    	panic("HANAMI_USER is not set")
    }
    if pw == "" {
    	panic("HANAMI_PW is not set")
    }
    if address == "" {
    	panic("HANAMI_ADDRESS is not set")
    }
    
    return hanami_sdk.RequestToken(address, user, pw, hanamictl_common.DisableTlsVerification)
}
