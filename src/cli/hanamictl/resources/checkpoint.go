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
    "os"
    "hanamictl/common"
    "github.com/spf13/cobra"
    "github.com/kitsudaiki/Hanami"
)

var checkpointHeader = []string{
    "uuid",
    "name",
    "visibility",
    "owner_id",
    "project_id",
    "created_at",
}

var getCheckpointCmd = &cobra.Command {
    Use:   "get CHECKPOINT_UUID",
    Short: "Get information of a specific checkpoint.",
    Args:  cobra.ExactArgs(1),
    Run:   func(cmd *cobra.Command, args []string) {
        token := Login()
        address := os.Getenv("HANAMI_ADDRESS")
        checkpointUuid := args[0]
        content, err := hanami_sdk.GetCheckpoint(address, token, checkpointUuid)
        if err != nil {
            fmt.Println(err)
            os.Exit(1)
        }
        hanamictl_common.ParseSingle(content, checkpointHeader)
    },
}

var listCheckpointCmd = &cobra.Command {
    Use:   "list",
    Short: "List all checkpoint.",
    Run:   func(cmd *cobra.Command, args []string) {
        token := Login()
        address := os.Getenv("HANAMI_ADDRESS")
        content, err := hanami_sdk.ListCheckpoint(address, token)
        if err != nil {
            fmt.Println(err)
            os.Exit(1)
        }
        hanamictl_common.ParseList(content, checkpointHeader)
    },
}

var deleteCheckpointCmd = &cobra.Command {
    Use:   "delete CHECKPOINT_UUID",
    Short: "Delete a specific checkpoint from the backend.",
    Args:  cobra.ExactArgs(1),
    Run:   func(cmd *cobra.Command, args []string) {
        token := Login()
        address := os.Getenv("HANAMI_ADDRESS")
        checkpointUuid := args[0]
        _, err := hanami_sdk.DeleteCheckpoint(address, token, checkpointUuid)
        if err != nil {
            fmt.Println(err)
            os.Exit(1)
        }
        fmt.Printf("successfully deleted checkpoint '%v'\n", checkpointUuid)
    },
}


var checkpointCmd = &cobra.Command {
    Use:   "checkpoint",
    Short: "Manage checkpoint.",
}


func Init_Checkpoint_Commands(rootCmd *cobra.Command) {
    rootCmd.AddCommand(checkpointCmd)

    checkpointCmd.AddCommand(getCheckpointCmd)

    checkpointCmd.AddCommand(listCheckpointCmd)

    checkpointCmd.AddCommand(deleteCheckpointCmd)
}
