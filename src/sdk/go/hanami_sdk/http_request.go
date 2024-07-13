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
    "encoding/json"
)

type RequestError struct {
    StatusCode int
    Err string
}

func (r *RequestError) Error() string {
    return fmt.Sprintf("status %d: err %v", r.StatusCode, r.Err)
}

func SendPost(address string, 
              token string, 
              path string, 
              jsonBody map[string]interface{}) (map[string]interface{}, error) {
    return sendGenericRequest(address, token, "POST", path, &jsonBody)
}

func SendPut(address string, 
             token string, 
             path string, 
             jsonBody map[string]interface{} ) (map[string]interface{}, error) {
    return sendGenericRequest(address, token, "PUT", path, &jsonBody)
}

func SendGet(address string, 
             token string, 
             path string, 
             vars map[string]string) (map[string]interface{}, error) {
    completePath := path + prepareVars(vars)
    return sendGenericRequest(address, token, "GET", completePath, nil)
}

func SendDelete(address string, 
                token string, 
                path string, 
                vars map[string]string) (map[string]interface{}, error) {
    completePath := path + prepareVars(vars)
    return sendGenericRequest(address, token, "DELETE", completePath, nil)
}

func prepareVars(vars map[string]string) string {
    if len(vars) > 0 {
        var pairs []string
        for key, value := range vars {
            pairs = append(pairs, fmt.Sprintf("%s=%s", key, value))
        }
        return fmt.Sprintf("?%s", strings.Join(pairs, "&"))
    }
    
    return ""
}

func sendGenericRequest(address string, 
                        token string, 
                        requestType string, 
                        path string, 
                        jsonBody *map[string]interface{}) (map[string]interface{}, error) {
    outputMap := map[string]interface{}{}
    jsonDataStr := ""
    if jsonBody != nil {
        jsonData, err := json.Marshal(jsonBody)
        if err != nil {
            return outputMap, err
        }
        jsonDataStr = string(jsonData)
    }

    // check if https or not
    if strings.Contains(address, "https") {
        http.DefaultTransport.(*http.Transport).TLSClientConfig = &tls.Config{InsecureSkipVerify: true}
    }

    // build uri
    var reqBody = strings.NewReader(jsonDataStr)
    // fmt.Printf("completePath: "+ completePath)
    completePath := fmt.Sprintf("%s/%s", address, path)
    req, err := http.NewRequest(requestType, completePath, reqBody)
    if err != nil {
        return outputMap, err
    }

    // add token to header
    if token != "" {
        req.Header.Set("X-Auth-Token", token)
    }

    // run request
    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        return outputMap, err
    }
    defer resp.Body.Close()

    // read data from response and convert into string
    bodyBytes, err := ioutil.ReadAll(resp.Body)
    if err != nil {
        return outputMap, err
    }
    bodyString := string(bodyBytes)

    // fmt.Printf("bodyString: " + bodyString + "\n")
    if resp.StatusCode != http.StatusOK {
        return outputMap, &RequestError{
            StatusCode: resp.StatusCode,
            Err:        bodyString,
        }
    }

    // parse result
    err = json.Unmarshal([]byte(bodyString), &outputMap)
    if err != nil {
        return outputMap, nil
    }

    return outputMap, nil
}
