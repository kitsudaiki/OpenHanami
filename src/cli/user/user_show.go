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

 package user_commands

import (
    "flag"
    "context"
    "fmt"
	"hanami_sdk"
	"../common"
	"github.com/google/subcommands"
)
  
type userShowCmd struct {
    name string
}

func (*userShowCmd) Name() string { 
    return "user_show" 
}

func (*userShowCmd) Synopsis() string { 
    return "Print information of a specific user." 
}

func (*userShowCmd) Usage() string {
    return `user_show --name <name>: Print information of the user with the name.`
}

func (p *userShowCmd) SetFlags(f *flag.FlagSet) {
    f.StringVar(&p.name, "name", "", "name of the user")
}

func (p *userShowCmd) Execute(_ context.Context, f *flag.FlagSet, _ ...interface{}) subcommands.ExitStatus {
	success, content := http_request.GetUser_Request(p.name)
	if success {
		hanami_cli_common.ParseSingle(content)
	} else {
		fmt.Println(content);
	}
    return subcommands.ExitSuccess
}

func Init_UserShow_command() {
	subcommands.Register(&userShowCmd{}, "user")
}
