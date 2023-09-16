/**
 * @file        user_show.go
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

 package documenation_commands

import (
	"encoding/base64"
    "os"
    "flag"
	"encoding/json"
    "context"
    "fmt"
	"hanami_sdk"
	"github.com/google/subcommands"
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

type generateRestApiDocumentationCmd struct {
}

func (*generateRestApiDocumentationCmd) Name() string { 
    return "generate_rest_api_documentation" 
}

func (*generateRestApiDocumentationCmd) Synopsis() string { 
    return "Print information of a specific user." 
}

func (*generateRestApiDocumentationCmd) Usage() string {
    return `generate_rest_api_documentation: .`
}

func (p *generateRestApiDocumentationCmd) SetFlags(f *flag.FlagSet) {
}

func (p *generateRestApiDocumentationCmd) Execute(_ context.Context, f *flag.FlagSet, _ ...interface{}) subcommands.ExitStatus {
	success, content := http_request.GetRestApiDocumentation_Request()

    

	if success {
		parseJson(content)
		outputMap := parseJson(content)
	
		base64String := outputMap["documentation"].(string)
		dec, err := base64.StdEncoding.DecodeString(base64String)
		if err != nil {
			panic(err)
		}

		f, err := os.Create("myfilename.pdf")
		if err != nil {
			panic(err)
		}
		defer f.Close()

		if _, err := f.Write(dec); err != nil {
			panic(err)
		}
		if err := f.Sync(); err != nil {
			panic(err)
		}

	} else {
		fmt.Println(content);
	}
    return subcommands.ExitSuccess
}

func Init_GenerateRestApiDocumentation_command() {
	subcommands.Register(&generateRestApiDocumentationCmd{}, "documentation")
}
