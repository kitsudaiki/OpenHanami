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

var (
    template       string
    checkpointName string
    checkpointUuid string
)

var createClusterCmd = &cobra.Command {
    Use:   "create CLUSTER_UUID",
    Short: "Create a new cluster.",
    Args:  cobra.ExactArgs(1),
    Run:   func(cmd *cobra.Command, args []string) {
        token := Login()
        address := os.Getenv("HANAMI_ADDRESS")
        clusterName := args[0]
        success, content := hanami_sdk.CreateCluster(address, token, clusterName, template)
        if success {
            hanamictl_common.ParseSingle(content)
        } else {
            fmt.Println(content)
        }
    },
}

var getClusterCmd = &cobra.Command {
    Use:   "get CLUSTER_UUID",
    Short: "Get information of a specific cluster.",
    Args:  cobra.ExactArgs(1),
    Run:   func(cmd *cobra.Command, args []string) {
        token := Login()
        address := os.Getenv("HANAMI_ADDRESS")
        clusterUuid := args[0]
        success, content := hanami_sdk.GetCluster(address, token, clusterUuid)
        if success {
            hanamictl_common.ParseSingle(content)
        } else {
            fmt.Println(content)
        }
    },
}

var listClusterCmd = &cobra.Command {
    Use:   "list",
    Short: "List all cluster.",
    Run:   func(cmd *cobra.Command, args []string) {
        token := Login()
        address := os.Getenv("HANAMI_ADDRESS")
        success, content := hanami_sdk.ListCluster(address, token)
        if success {
            hanamictl_common.ParseList(content)
        } else {
            fmt.Println(content)
        }
    },
}

var deleteClusterCmd = &cobra.Command {
    Use:   "delete CLUSTER_UUID",
    Short: "Delete a specific cluster from the backend.",
    Args:  cobra.ExactArgs(1),
    Run:   func(cmd *cobra.Command, args []string) {
        token := Login()
        address := os.Getenv("HANAMI_ADDRESS")
        clusterUuid := args[0]
        success, content := hanami_sdk.DeleteCluster(address, token, clusterUuid)
        if success {
            fmt.Println("successfully deleted cluster '%s'", clusterUuid)
        } else {
            fmt.Println(content)
        }
    },
}

var saveClusterCmd = &cobra.Command {
    Use:   "save CLUSTER_UUID",
    Short: "Save cluster as checkpoint.",
    Args:  cobra.ExactArgs(1),
    Run:   func(cmd *cobra.Command, args []string) {
        token := Login()
        address := os.Getenv("HANAMI_ADDRESS")
        clusterUuid := args[0]
        success, content := hanami_sdk.SaveCluster(address, token, clusterUuid, checkpointName)
        if success {
            hanamictl_common.ParseSingle(content)
        } else {
            fmt.Println(content)
        }
    },
}

var restoreClusterCmd = &cobra.Command {
    Use:   "restore CLUSTER_UUID",
    Short: "Restore cluster from checkpoint.",
    Args:  cobra.ExactArgs(1),
    Run:   func(cmd *cobra.Command, args []string) {
        token := Login()
        address := os.Getenv("HANAMI_ADDRESS")
        clusterUuid := args[0]
        success, content := hanami_sdk.RestoreCluster(address, token, clusterUuid, checkpointUuid)
        if success {
            hanamictl_common.ParseSingle(content)
        } else {
            fmt.Println(content)
        }
    },
}


var clusterCmd = &cobra.Command {
    Use:   "cluster",
    Short: "Manage cluster.",
}


func Init_Cluster_Commands(rootCmd *cobra.Command) {
    rootCmd.AddCommand(clusterCmd)

    clusterCmd.AddCommand(createClusterCmd)
    createClusterCmd.Flags().StringVarP(&template, "template", "t", "", "Cluster Template (mandatory)")
    createClusterCmd.MarkFlagRequired("template")

    clusterCmd.AddCommand(getClusterCmd)

    clusterCmd.AddCommand(listClusterCmd)

    clusterCmd.AddCommand(deleteClusterCmd)

    clusterCmd.AddCommand(saveClusterCmd)
    saveClusterCmd.Flags().StringVarP(&checkpointName, "name", "n", "", "Checkpoint name (mandatory)")
    saveClusterCmd.MarkFlagRequired("name")

    clusterCmd.AddCommand(restoreClusterCmd)
    restoreClusterCmd.Flags().StringVarP(&checkpointUuid, "checkpoint", "t", "", "Checkpoint UUID (mandatory)")
    restoreClusterCmd.MarkFlagRequired("checkpoint")
}

