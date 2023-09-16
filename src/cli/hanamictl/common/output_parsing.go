/**
 * @file        output_parsing.go
  *
 * @author      Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright   Apache License Version 2.0
 *
 *      Copyright 2021 Tobias Anker
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

 package hanami_cli_common

import (
    "fmt"
    "os"
    "strings"
    "encoding/json"
    "github.com/olekukonko/tablewriter"
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

func ParseSingle(input string) {
    outputMap := parseJson(input)
    table := tablewriter.NewWriter(os.Stdout)

    for k, v := range outputMap { 
        lineData := []string{}
        lineData = append(lineData, strings.ToUpper(k))
        lineData = append(lineData, fmt.Sprintf("%v", v))
        table.Append(lineData)
    }
    
    table.SetRowLine(true)
    table.Render()
}

func ParseList(input string) {
    outputMap := parseJson(input)
    table := tablewriter.NewWriter(os.Stdout)

    // add header to table
    headerData := []string{}
    headerArray := outputMap["header"].([]interface{})
    for _, val := range headerArray {
        str := fmt.Sprintf("%v", val)
        headerData = append(headerData, str)
    }
    table.SetHeader(headerData)
    
    // add body to table
    bodyArray := outputMap["body"].([]interface{})
    for _, line := range bodyArray {
        lineData := []string{}
        for _, val := range line.([]interface{}) {
            if strVal, ok := val.(string); ok {
                lineData = append(lineData, strVal)
            } else {
                str := fmt.Sprintf("%v", val)
                lineData = append(lineData, str)
            }
        }        
        table.Append(lineData)
    }
    
    table.SetRowLine(true)
    table.Render()
}

