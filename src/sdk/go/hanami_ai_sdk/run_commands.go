/**
 * @file        run_commands.go
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

 package http_request

import (
    "fmt"
)

func RunLearn_Request(number_of_inputs_per_cycle string, number_of_outputs_per_cycle string, number_of_cycles string, cluster_uuid string, inputs string, label string) (bool, string) {
	path := "control/v1/io"
	vars := ""
	jsonBody := fmt.Sprintf("{\"number_of_inputs_per_cycle\":%s, \"number_of_outputs_per_cycle\":%s, \"number_of_cycles\":%s, \"cluster_uuid\":\"%s\", \"inputs\":\"%s\", \"label\":\"%s\"}", 
	                        number_of_inputs_per_cycle, 
							number_of_outputs_per_cycle, 
							number_of_cycles, 
							cluster_uuid, 
							inputs, 
							label)
    return SendPost_Request(path, vars, jsonBody)
}

func RunAsk_Request(number_of_inputs_per_cycle string, number_of_cycles string, cluster_uuid string, inputs string) (bool, string) {
	path := "control/v1/io"
	vars := ""
	jsonBody := fmt.Sprintf("{\"number_of_inputs_per_cycle\":%s, \"number_of_cycles\":%s, \"cluster_uuid\":\"%s\", \"inputs\":\"%s\"}", 
	                        number_of_inputs_per_cycle, 
				    		number_of_cycles, 
					    	cluster_uuid, 
					    	inputs)
	return SendPost_Request(path, vars, jsonBody)
}
