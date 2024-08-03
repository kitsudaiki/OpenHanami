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

package hanamictl_common

import (
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"strconv"
	"strings"

	"github.com/olekukonko/tablewriter"
)

var PrintAsJson bool = false
var DisableTlsVerification bool = false

func PrintSingle(input map[string]interface{}, outputFields []string) {
	if PrintAsJson {
		jsonData, _ := json.MarshalIndent(input, "", "    ")
		fmt.Println(string(jsonData))
		return
	}

	table := tablewriter.NewWriter(os.Stdout)

	for _, element := range outputFields {
		v := input[element]
		lineData := []string{}
		lineData = append(lineData, strings.ToUpper(strings.ReplaceAll(element, "_", " ")))
		val := fmt.Sprintf("%v", v)
		if reflect.ValueOf(v).Kind() == reflect.Map {
			jsonData, _ := json.Marshal(v)
			lineData = append(lineData, string(jsonData))
		} else {
			lineData = append(lineData, val)
		}
		table.Append(lineData)
	}

	table.SetRowLine(false)
	table.SetAlignment(tablewriter.ALIGN_LEFT)
	// table.EnableBorder(false)
	table.Render()
}

func searchInHeader(headerArray []interface{}, name string) int {
	for index, val := range headerArray {
		str := fmt.Sprintf("%v", val)
		if str == name {
			return index
		}
	}

	return -1
}

func PrintList(input map[string]interface{}, outputFields []string) {
	if PrintAsJson {
		jsonData, _ := json.MarshalIndent(input, "", "    ")
		fmt.Println(string(jsonData))
		return
	}

	table := tablewriter.NewWriter(os.Stdout)
	headerArray := input["header"].([]interface{})
	bodyArray := input["body"].([]interface{})

	// fill and add table header
	headerPositions := []int{}
	headerData := []string{}
	for _, element := range outputFields {
		pos := searchInHeader(headerArray, element)
		if pos == -1 {
			continue
		}

		str := fmt.Sprintf("%v", headerArray[pos])
		headerData = append(headerData, str)
		headerPositions = append(headerPositions, pos)
	}
	table.SetHeader(headerData)

	// fill and add body to table
	for _, line := range bodyArray {
		lineData := []string{}
		for _, pos := range headerPositions {
			val := line.([]interface{})[pos]
			if strVal, ok := val.(string); ok {
				lineData = append(lineData, strVal)
			} else {
				str := fmt.Sprintf("%v", val)
				lineData = append(lineData, str)
			}
		}
		table.Append(lineData)
	}

	table.SetRowLine(false)
	table.SetAlignment(tablewriter.ALIGN_LEFT)
	table.Render()
}

func PrintValueList(data []interface{}, offset int) {
	if PrintAsJson {
		jsonData, _ := json.MarshalIndent(data, "", "    ")
		fmt.Println(string(jsonData))
		return
	}

	table := tablewriter.NewWriter(os.Stdout)

	// fill and add table header
	headerData := []string{}
	headerData = append(headerData, "")
	for i := range len(data[0].([]interface{})) {
		headerData = append(headerData, strconv.Itoa(i))
	}
	table.SetHeader(headerData)

	// fill and add body to table
	for i, line := range data {
		lineData := []string{}
		lineData = append(lineData, fmt.Sprintf("%d", (offset+i)))
		for _, val := range line.([]interface{}) {
			lineData = append(lineData, fmt.Sprintf("%f", val))
		}
		table.Append(lineData)
	}

	table.SetRowLine(false)
	table.SetAlignment(tablewriter.ALIGN_LEFT)
	table.Render()
}
