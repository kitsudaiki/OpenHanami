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

var hostHeader = []string{
    "uuid",
    "type",
}

var listHostsCmd = &cobra.Command {
    Use:   "list",
    Short: "List all logical hosts.",
    Run:   func(cmd *cobra.Command, args []string) {
        token := Login()
        address := os.Getenv("HANAMI_ADDRESS")
        content, err := hanami_sdk.ListHosts(address, token)
        if err == nil {
            hanamictl_common.ParseList(content, hostHeader)
        } else {
            fmt.Println(err)
            os.Exit(1)
        }
    },
}

var hostsCmd = &cobra.Command {
    Use:   "host",
    Short: "Manage hosts.",
}

func Init_Host_Commands(rootCmd *cobra.Command) {
    rootCmd.AddCommand(hostsCmd)

    hostsCmd.AddCommand(listHostsCmd)
}
