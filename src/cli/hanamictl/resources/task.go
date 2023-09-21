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

var (
    taskName    string
    clusterUuid string
    datasetUuid string
    taskType    string
)

var createTaskCmd = &cobra.Command {
    Use:   "create TASK_ID",
    Short: "Create a new task.",
    Args:  cobra.ExactArgs(1),
    Run:   func(cmd *cobra.Command, args []string) {
        taskName := args[0]
        success, content := hanami_sdk.CreateTask(taskName, taskType, clusterUuid, datasetUuid)
        if success {
            hanamictl_common.ParseSingle(content)
        } else {
            fmt.Println(content)
        }
    },
}

var getTaskCmd = &cobra.Command {
    Use:   "get TASK_ID",
    Short: "Get information of a specific task.",
    Args:  cobra.ExactArgs(1),
    Run:   func(cmd *cobra.Command, args []string) {
        taskId := args[0]
        success, content := hanami_sdk.GetTask(taskId, clusterUuid)
        if success {
            hanamictl_common.ParseSingle(content)
        } else {
            fmt.Println(content)
        }
    },
}

var listTaskCmd = &cobra.Command {
    Use:   "list",
    Short: "List all task.",
    Run:   func(cmd *cobra.Command, args []string) {
        success, content := hanami_sdk.ListTask(clusterUuid)
        if success {
            hanamictl_common.ParseList(content)
        } else {
            fmt.Println(content)
        }
    },
}

var deleteTaskCmd = &cobra.Command {
    Use:   "delete TASK_ID",
    Short: "Delete a specific task from the backend.",
    Args:  cobra.ExactArgs(1),
    Run:   func(cmd *cobra.Command, args []string) {
        taskId := args[0]
        success, content := hanami_sdk.DeleteTask(taskId, clusterUuid)
        if success {
            fmt.Println("successfully deleted task '%s'", taskId)
        } else {
            fmt.Println(content)
        }
    },
}


var taskCmd = &cobra.Command {
    Use:   "task",
    Short: "Manage task.",
}


func Init_Task_Commands(rootCmd *cobra.Command) {
    rootCmd.AddCommand(taskCmd)

    taskCmd.AddCommand(createTaskCmd)
    createTaskCmd.Flags().StringVarP(&clusterUuid, "cluster", "c", "", "Cluster UUID (mandatory)")
    createTaskCmd.Flags().StringVarP(&clusterUuid, "dataset", "d", "", "Data-Set UUID (mandatory)")
    createTaskCmd.Flags().StringVarP(&clusterUuid, "type", "t", "", "Task type (mandatory)")
    createTaskCmd.MarkFlagRequired("cluster")
    createTaskCmd.MarkFlagRequired("dataset")
    createTaskCmd.MarkFlagRequired("type")

    taskCmd.AddCommand(getTaskCmd)
    getTaskCmd.Flags().StringVarP(&clusterUuid, "cluster", "c", "", "Cluster UUID (mandatory)")
    getTaskCmd.MarkFlagRequired("cluster")

    taskCmd.AddCommand(listTaskCmd)
    listTaskCmd.Flags().StringVarP(&clusterUuid, "cluster", "c", "", "Cluster UUID (mandatory)")
    listTaskCmd.MarkFlagRequired("cluster")

    taskCmd.AddCommand(deleteTaskCmd)
    deleteTaskCmd.Flags().StringVarP(&clusterUuid, "cluster", "c", "", "Cluster UUID (mandatory)")
    deleteTaskCmd.MarkFlagRequired("cluster")
}
