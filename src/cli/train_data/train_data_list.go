/**
 * @file        train_data_list.go
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
  
type trainDataListCmd struct {
}

func (*trainDataListCmd) Name() string { 
    return "train_data_list" 
}

func (*trainDataListCmd) Synopsis() string { 
    return "Print args to stdout." 
}

func (*trainDataListCmd) Usage() string {
    return `train_data_list <some text>: Print args to stdout.`
}

func (p *trainDataListCmd) SetFlags(f *flag.FlagSet) {
}

func (p *trainDataListCmd) Execute(_ context.Context, f *flag.FlagSet, _ ...interface{}) subcommands.ExitStatus {
	success, content := http_request.ListTrainData_Request()
	if success {
		hanami_cli_common.ParseList(content)
	} else {
		fmt.Println(content);
	}
    return subcommands.ExitSuccess
}

func Init_TrainDataList_command() {
	subcommands.Register(&trainDataListCmd{}, "train_data")
}
