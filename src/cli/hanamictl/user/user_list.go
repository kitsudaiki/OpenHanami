/**
 * @file        user_list.go
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
    "fmt"
	"github.com/kitsudaiki/Hanami"
	"hanamictl/common"
    "github.com/spf13/cobra"
)

var listCmd = &cobra.Command{
    Use:   "list",
    Short: "List resources of one type",
}

var userListCmd = &cobra.Command{
    Use:   "user",
    Short: "List all users.",
}

func userListRun(cmd *cobra.Command, args []string) {
	success, content := http_request.ListUser_Request()
	if success {
		hanami_cli_common.ParseList(content)
	} else {
		fmt.Println(content);
	}
}

func Init_UserList_command(rootCmd *cobra.Command) {
	rootCmd.AddCommand(listCmd)
    listCmd.AddCommand(userListCmd)

    userListCmd.Run = userListRun
}
