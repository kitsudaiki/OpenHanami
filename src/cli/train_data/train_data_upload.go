/**
 * @file        train_data_upload.go
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

 package train_data_commands

import (
    "flag"
    "context"
    "fmt"
	"hanami_sdk"
	"../common"
	"github.com/google/subcommands"
)
  
type trainDataCreateCmd struct {
    name string
	dataType string
	dataPath string
}

func (*trainDataCreateCmd) Name() string { 
    return "train_data_create" 
}

func (*trainDataCreateCmd) Synopsis() string { 
    return "Upload new train-data." 
}

func (*trainDataCreateCmd) Usage() string {
    return `train_data_create --name <name> --type <type> --path <path>: Upload train-data-file.`
}

func (p *trainDataCreateCmd) SetFlags(f *flag.FlagSet) {
    f.StringVar(&p.name, "name", "", "Name of the data for better later identification.")
	f.StringVar(&p.dataType, "type", "", "Type of the data.")
    f.StringVar(&p.dataPath, "path", "", "Local path to the file to upload")
}

func (p *trainDataCreateCmd) Execute(_ context.Context, f *flag.FlagSet, _ ...interface{}) subcommands.ExitStatus {
	success, content := http_request.UploadTrainData_Request(p.name, p.dataType, p.dataPath)
	if success {
		hanami_cli_common.ParseSingle(content)
	} else {
		fmt.Println(content);
	}
    return subcommands.ExitSuccess
}

func Init_TrainDataCreate_command() {
	subcommands.Register(&trainDataCreateCmd{}, "train_data")
}
