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

package hanami_resources

import (
    "fmt"
    
    "hanamictl/common"
    "github.com/spf13/cobra"
    "github.com/kitsudaiki/Hanami"
)


var getRequestResultCmd = &cobra.Command {
    Use:   "get REQUEST_RESULT_UUID",
    Short: "Get information of a specific request result.",
    Args:  cobra.ExactArgs(1),
    Run:   func(cmd *cobra.Command, args []string) {
        requestResultUuid := args[0]
        success, content := hanami_sdk.GetRequestResult(requestResultUuid)
        if success {
            hanamictl_common.ParseSingle(content)
        } else {
            fmt.Println(content)
        }
    },
}

var listRequestResultCmd = &cobra.Command {
    Use:   "list",
    Short: "List all request result.",
    Run:   func(cmd *cobra.Command, args []string) {
        success, content := hanami_sdk.ListRequestResult()
        if success {
            hanamictl_common.ParseList(content)
        } else {
            fmt.Println(content)
        }
    },
}

var deleteRequestResultCmd = &cobra.Command {
    Use:   "delete REQUEST_RESULT_UUID",
    Short: "Delete a specific request result from the backend.",
    Args:  cobra.ExactArgs(1),
    Run:   func(cmd *cobra.Command, args []string) {
        requestResultUuid := args[0]
        success, content := hanami_sdk.DeleteRequestResult(requestResultUuid)
        if success {
            fmt.Println("successfully deleted request result '%s'", requestResultUuid)
        } else {
            fmt.Println(content)
        }
    },
}


var requestResultCmd = &cobra.Command {
    Use:   "request result",
    Short: "Manage request result.",
}


func Init_RequestResult_Commands(rootCmd *cobra.Command) {
    rootCmd.AddCommand(requestResultCmd)

    requestResultCmd.AddCommand(getRequestResultCmd)

    requestResultCmd.AddCommand(listRequestResultCmd)

    requestResultCmd.AddCommand(deleteRequestResultCmd)
}
