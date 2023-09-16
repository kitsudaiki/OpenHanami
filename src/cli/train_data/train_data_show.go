/**
 * @file        train_data_show.go
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
  
type trainDataShowCmd struct {
    uuid string
	with_data bool
}

func (*trainDataShowCmd) Name() string { 
    return "train_data_show" 
}

func (*trainDataShowCmd) Synopsis() string { 
    return "Print args to stdout." 
}

func (*trainDataShowCmd) Usage() string {
    return `train_data_show --uuid <uuid> [--with_data]: Print args to stdout.`
}

func (p *trainDataShowCmd) SetFlags(f *flag.FlagSet) {
    f.StringVar(&p.uuid, "uuid", "", "uuid of the train-data set for identification")
	f.BoolVar(&p.with_data, "with_data", false, "has to be set if also the content of the train-data file should be returned")
}

func (p *trainDataShowCmd) Execute(_ context.Context, f *flag.FlagSet, _ ...interface{}) subcommands.ExitStatus {
	success, content := http_request.GetTrainData_Request(p.uuid, p.with_data)
	if success {
		hanami_cli_common.ParseSingle(content)
	} else {
		fmt.Println(content);
	}
    return subcommands.ExitSuccess
}

func Init_TrainDataShow_command() {
	subcommands.Register(&trainDataShowCmd{}, "train_data")
}
