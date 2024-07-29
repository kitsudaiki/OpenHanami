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

package main

import (
    "fmt"
    "github.com/spf13/cobra"
    "os"
    "hanamictl/resources"
    "hanamictl/common"
)

var rootCmd = &cobra.Command{Use: "hanamictl"}

func init() {
    rootCmd.PersistentFlags().BoolVarP(&hanamictl_common.PrintAsJson, "json_output", "j", false, "Return output as json")
    rootCmd.PersistentFlags().BoolVar(&hanamictl_common.DisableTlsVerification, "insecure", false, "Disable the TLS-verification")

    hanami_resources.Init_User_Commands(rootCmd);
    hanami_resources.Init_Project_Commands(rootCmd);
    hanami_resources.Init_Checkpoint_Commands(rootCmd);
    hanami_resources.Init_Task_Commands(rootCmd);
    hanami_resources.Init_Cluster_Commands(rootCmd);
    hanami_resources.Init_Dataset_Commands(rootCmd);
    hanami_resources.Init_Host_Commands(rootCmd)
}

func main() {
    if err := rootCmd.Execute(); err != nil {
        fmt.Println(err)
        os.Exit(1)
    }
}
