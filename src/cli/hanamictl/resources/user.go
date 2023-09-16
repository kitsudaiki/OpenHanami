/**
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

package hanami_user_commands

import (
    "fmt"
	"hanamictl/common"
    "github.com/spf13/cobra"
	"github.com/kitsudaiki/Hanami"
)


func userListRun(cmd *cobra.Command, args []string) {
	success, content := http_request.ListUser_Request()
	if success {
		hanamictl_common.ParseList(content)
	} else {
		fmt.Println(content);
	}
}

func userGetRun(cmd *cobra.Command, args []string) {
	userId := args[0]

	success, content := http_request.GetUser_Request(userId)
	if success {
		hanamictl_common.ParseSingle(content)
	} else {
		fmt.Println(content);
	}
}


var listUserCmd = &cobra.Command {
    Use:   "list",
    Short: "List all user.",
	Run: userListRun,
}

var getUserCmd = &cobra.Command {
    Use:   "get <USER_ID>",
    Short: "Get information of a specific user.",
	Args:  cobra.ExactArgs(1),
	Run: userGetRun,
}

var userCmd = &cobra.Command {
    Use:   "user",
    Short: "Manage user.",
}


func Init_User_Commands(rootCmd *cobra.Command) {
	rootCmd.AddCommand(userCmd)
    userCmd.AddCommand(listUserCmd)
    userCmd.AddCommand(getUserCmd)
}
