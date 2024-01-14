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
    "net/http"
    "io/ioutil"
    "crypto/tls"
    "strings"
)

func SendGet(address string, token string, path string, vars string) (bool, string) {
    return sendRequest(address, token, "GET", path, vars, "")
}

func SendPost(address string, token string, path string, vars string, jsonBody string) (bool, string) {
    return sendRequest(address, token, "POST", path, vars, jsonBody)
}

func SendPut(address string, token string, path string, vars string, jsonBody string) (bool, string) {
    return sendRequest(address, token, "PUT", path, vars, jsonBody)
}

func SendDelete(address string, token string, path string, vars string) (bool, string) {
    return sendRequest(address, token, "DELETE", path, vars, "")
}

func sendRequest(address string, token string, requestType string, path string, vars string, jsonBody string) (bool, string) {
    completePath := path
    if vars != "" {
        completePath += fmt.Sprintf("?%s", vars)
    }
    
    return sendGenericRequest(address, token, requestType, completePath, jsonBody)
}

func sendGenericRequest(address string, token string, requestType string, path string, jsonBody string) (bool, string) {
    // check if https or not
    if strings.Contains(address, "https") {
        http.DefaultTransport.(*http.Transport).TLSClientConfig = &tls.Config{InsecureSkipVerify: true}
    }

    // build uri
    var reqBody = strings.NewReader(jsonBody)
    // fmt.Printf("completePath: "+ completePath)
    completePath := fmt.Sprintf("%s/%s", address, path)
    req, err := http.NewRequest(requestType, completePath, reqBody)
    if err != nil {
        panic(err)
    }

    // add token to header
    if token != "" {
        req.Header.Set("X-Auth-Token", token)
    }

    // run request
    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        panic(err)
    }
    defer resp.Body.Close()

    // read data from response and convert into string
    bodyBytes, err := ioutil.ReadAll(resp.Body)
    if err != nil {
        return false, ""
    }
    bodyString := string(bodyBytes)

    // fmt.Printf("bodyString: " + bodyString + "\n")

    var ok = resp.StatusCode == http.StatusOK
    return ok, bodyString
}
