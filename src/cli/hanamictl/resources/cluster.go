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
    templatePath   string
    checkpointName string
    checkpointUuid string
)

var clusterHeader = []string{
    "uuid",
    "name",
    "visibility",
    "owner_id",
    "project_id",
    "created_at",
}

var createClusterCmd = &cobra.Command {
    Use:   "create -t TEMPLATE_PATH NAME",
    Short: "Create a new cluster.",
    Args:  cobra.ExactArgs(1),
    Run:   func(cmd *cobra.Command, args []string) {
        token := Login()
        address := os.Getenv("HANAMI_ADDRESS")
        clusterName := args[0]
        templateContent, err := os.ReadFile(templatePath)
        if err != nil {
            fmt.Println(err)
            os.Exit(1)
        }
        content, err := hanami_sdk.CreateCluster(address, token, clusterName, string(templateContent))
        if err != nil {
            fmt.Println(err)
            os.Exit(1)
        }
        hanamictl_common.ParseSingle(content, clusterHeader)
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
        content, err := hanami_sdk.GetCluster(address, token, clusterUuid)
        if err != nil {
            fmt.Println(err)
            os.Exit(1)
        }
        hanamictl_common.ParseSingle(content, clusterHeader)
    },
}

var listClusterCmd = &cobra.Command {
    Use:   "list",
    Short: "List all cluster.",
    Run:   func(cmd *cobra.Command, args []string) {
        token := Login()
        address := os.Getenv("HANAMI_ADDRESS")
        content, err := hanami_sdk.ListCluster(address, token)
        if err != nil {
            fmt.Println(err)
            os.Exit(1)
        }
        hanamictl_common.ParseList(content, clusterHeader)
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
        _, err := hanami_sdk.DeleteCluster(address, token, clusterUuid)
        if err != nil {
            fmt.Println(err)
            os.Exit(1)
        }
        fmt.Printf("successfully deleted cluster '%v'\n", clusterUuid)
    },
}

var saveClusterCmd = &cobra.Command {
    Use:   "save -n CHECKPOINT_NAME CLUSTER_UUID",
    Short: "Save cluster as checkpoint.",
    Args:  cobra.ExactArgs(1),
    Run:   func(cmd *cobra.Command, args []string) {
        token := Login()
        address := os.Getenv("HANAMI_ADDRESS")
        clusterUuid := args[0]
        content, err := hanami_sdk.SaveCluster(address, token, clusterUuid, checkpointName)
        if err != nil {
            fmt.Println(err)
            os.Exit(1)
        }
        hanamictl_common.ParseSingle(content, clusterHeader)
    },
}

var restoreClusterCmd = &cobra.Command {
    Use:   "restore -c CHECKPOINT_UUID CLUSTER_UUID",
    Short: "Restore cluster from checkpoint.",
    Args:  cobra.ExactArgs(1),
    Run:   func(cmd *cobra.Command, args []string) {
        token := Login()
        address := os.Getenv("HANAMI_ADDRESS")
        clusterUuid := args[0]
        content, err := hanami_sdk.RestoreCluster(address, token, clusterUuid, checkpointUuid)
        if err != nil {
            fmt.Println(err)
            os.Exit(1)
        }
        hanamictl_common.ParseSingle(content, clusterHeader)
    },
}


var clusterCmd = &cobra.Command {
    Use:   "cluster",
    Short: "Manage cluster.",
}


func Init_Cluster_Commands(rootCmd *cobra.Command) {
    rootCmd.AddCommand(clusterCmd)

    clusterCmd.AddCommand(createClusterCmd)
    createClusterCmd.Flags().StringVarP(&templatePath, "template", "t", "", "Cluster Template (mandatory)")
    createClusterCmd.MarkFlagRequired("template")

    clusterCmd.AddCommand(getClusterCmd)

    clusterCmd.AddCommand(listClusterCmd)

    clusterCmd.AddCommand(deleteClusterCmd)

    clusterCmd.AddCommand(saveClusterCmd)
    saveClusterCmd.Flags().StringVarP(&checkpointName, "name", "n", "", "Checkpoint name (mandatory)")
    saveClusterCmd.MarkFlagRequired("name")

    clusterCmd.AddCommand(restoreClusterCmd)
    restoreClusterCmd.Flags().StringVarP(&checkpointUuid, "checkpoint", "c", "", "Checkpoint UUID (mandatory)")
    restoreClusterCmd.MarkFlagRequired("checkpoint")
}

