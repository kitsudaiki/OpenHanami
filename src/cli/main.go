/**
 * @file        main.go
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

 package main

import (
    "flag"
    "context"
    "os"
    "./user"
    "./train_data"
    "./documentation"
    "github.com/google/subcommands"
)

func main() {
    subcommands.Register(subcommands.HelpCommand(), "")
    subcommands.Register(subcommands.FlagsCommand(), "")
    subcommands.Register(subcommands.CommandsCommand(), "")

    user_commands.Init_UserShow_command();
    user_commands.Init_UserList_command();

    train_data_commands.Init_TrainDataCreate_command();
    train_data_commands.Init_TrainDataShow_command();
    train_data_commands.Init_TrainDataList_command();

    documenation_commands.Init_GenerateRestApiDocumentation_command();

    flag.Parse()
    ctx := context.Background()
    os.Exit(int(subcommands.Execute(ctx)))
}
