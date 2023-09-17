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
	"strconv"
    "os"
    "encoding/json"
)

func SendGet_Request(path string, vars string) (bool, string) {
    return sendHanamiRequest("GET", path, vars, "")
}

func SendPost_Request(path string, vars string, jsonBody string) (bool, string) {
    return sendHanamiRequest("POST", path, vars, jsonBody)
}

func SendPut_Request(path string, vars string, jsonBody string) (bool, string) {
    return sendHanamiRequest("PUT", path, vars, jsonBody)
}

func SendDelete_Request(path string, vars string) (bool, string) {
    return sendHanamiRequest("DELETE", path, vars, "")
}


func sendHanamiRequest(requestType string, path string, vars string, jsonBody string) (bool, string){
    var token = os.Getenv("HANAMI_TOKEN")

    // request token, if no one exist within the environment-variables
    if token == "" {
        success := requestToken()
        if success == false {
            return false, "ACCESS DENIED!"
        }
    }
    
    // make request
    token = os.Getenv("HANAMI_TOKEN")
    success, content := sendRequest(requestType, token, path, vars, jsonBody)

    // hande expired token
    if success && content == "Token is expired" {
        success := requestToken()
        if success == false {
            return false, "ACCESS DENIED!"
        }

        // make new request with new token
        token = os.Getenv("HANAMI_TOKEN")

        return sendRequest(requestType, token, path, vars, jsonBody)
    }

    return success, content
}


func parseJson(input string) map[string]interface{} {
    // parse json and fill into map
    outputMap := map[string]interface{}{}
    err := json.Unmarshal([]byte(input), &outputMap)
    if err != nil {
        panic(err)
    }

    return outputMap
}


func requestToken() bool {
    var user = os.Getenv("HANAMI_USER")
	var pw = os.Getenv("HANAMI_PW")

    path := fmt.Sprintf("control/v1/token?name=%s&pw=%s", user, pw)

    success, content := sendGenericRequest("GET", "", path, "")
    if success == false {
        return false
    }

    outputMap := parseJson(content)
    token := outputMap["token"].(string)
    os.Setenv("HANAMI_TOKEN", token)

    return true
}

func sendRequest(requestType string, token string, path string, vars string, jsonBody string) (bool, string) {
    completePath := path
    if vars != "" {
        completePath += fmt.Sprintf("?%s", vars)
    }
    
    return sendGenericRequest(requestType, token, completePath, jsonBody)
}


func sendGenericRequest(requestType string, token string, path string, jsonBody string) (bool, string) {
    // read environment-variables
	var address = os.Getenv("HANAMI_ADDRESS")
	port, err := strconv.Atoi(os.Getenv("HANAMI_PORT"))
    if err != nil {
        return false, "err"
    }

    // check if https or not
	if strings.Contains(address, "https") {
		http.DefaultTransport.(*http.Transport).TLSClientConfig = &tls.Config{InsecureSkipVerify: true}
	}

    // build uri
    var reqBody = strings.NewReader(jsonBody)
    // fmt.Printf("completePath: "+ completePath)
    completePath := fmt.Sprintf("%s:%d/%s", address, port, path)
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
