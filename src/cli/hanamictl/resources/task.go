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
	hanamictl_common "hanamictl/common"
	"os"

	hanami_sdk "github.com/kitsudaiki/OpenHanami"
	"github.com/spf13/cobra"
)

var (
	clusterUuid string
	inputData   []string
	outputData  []string
	timeLength  int
)

var taskHeader = []string{
	"uuid",
	"state",
	"current_cycle",
	"total_number_of_cycles",
	"queue_timestamp",
	"start_timestamp",
	"end_timestamp",
}

var createTrainTaskCmd = &cobra.Command{
	Use:   "train -i HEXAGON_NAME:DATASET_UUID -c CLUSTER_UUID TASK_NAME",
	Short: "Create a new train task.",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		token := Login()
		address := os.Getenv("HANAMI_ADDRESS")
		taskName := args[0]
		content, err := hanami_sdk.CreateTrainTask(address, token, taskName, clusterUuid, inputData, outputData, timeLength, hanamictl_common.DisableTlsVerification)
		if err == nil {
			hanamictl_common.PrintSingle(content, taskHeader)
		} else {
			fmt.Println(err)
			os.Exit(1)
		}
	},
}

var createRequestTaskCmd = &cobra.Command{
	Use:   "request -i HEXAGON_NAME:DATASET_UUID -c CLUSTER_UUID TASK_NAME",
	Short: "Create a new request task.",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		token := Login()
		address := os.Getenv("HANAMI_ADDRESS")
		taskName := args[0]
		content, err := hanami_sdk.CreateRequestTask(address, token, taskName, clusterUuid, inputData, outputData, timeLength, hanamictl_common.DisableTlsVerification)
		if err == nil {
			hanamictl_common.PrintSingle(content, taskHeader)
		} else {
			fmt.Println(err)
			os.Exit(1)
		}
	},
}

var getTaskCmd = &cobra.Command{
	Use:   "get TASK_ID",
	Short: "Get information of a specific task.",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		token := Login()
		address := os.Getenv("HANAMI_ADDRESS")
		taskId := args[0]
		content, err := hanami_sdk.GetTask(address, token, taskId, clusterUuid, hanamictl_common.DisableTlsVerification)
		if err == nil {
			hanamictl_common.PrintSingle(content, taskHeader)
		} else {
			fmt.Println(err)
			os.Exit(1)
		}
	},
}

var listTaskCmd = &cobra.Command{
	Use:   "list",
	Short: "List all task.",
	Run: func(cmd *cobra.Command, args []string) {
		token := Login()
		address := os.Getenv("HANAMI_ADDRESS")
		content, err := hanami_sdk.ListTask(address, token, clusterUuid, hanamictl_common.DisableTlsVerification)
		if err == nil {
			hanamictl_common.PrintList(content, taskHeader)
		} else {
			fmt.Println(err)
			os.Exit(1)
		}
	},
}

var deleteTaskCmd = &cobra.Command{
	Use:   "delete TASK_ID",
	Short: "Delete a specific task from the backend.",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		token := Login()
		address := os.Getenv("HANAMI_ADDRESS")
		taskUuid := args[0]
		_, err := hanami_sdk.DeleteTask(address, token, taskUuid, clusterUuid, hanamictl_common.DisableTlsVerification)
		if err == nil {
			fmt.Printf("successfully deleted task '%v'\n", taskUuid)
		} else {
			fmt.Println(err)
			os.Exit(1)
		}
	},
}

var taskCmd = &cobra.Command{
	Use:   "task",
	Short: "Manage task.",
}

var createTaskCmd = &cobra.Command{
	Use:   "create",
	Short: "Create new task.",
}

func Init_Task_Commands(rootCmd *cobra.Command) {
	rootCmd.AddCommand(taskCmd)

	taskCmd.AddCommand(createTaskCmd)

	createTaskCmd.AddCommand(createTrainTaskCmd)
	createTrainTaskCmd.Flags().StringSliceVarP(&inputData, "input", "i", []string{}, "Cluster input, which are paris of '-i <HEXAGON_NAME>:<DATASET_UUID>' (mandatory)")
	createTrainTaskCmd.Flags().StringSliceVarP(&outputData, "output", "o", []string{}, "Cluster outputs, which are paris of '-o <HEXAGON_NAME>:<DATASET_UUID>' (mandatory)")
	createTrainTaskCmd.Flags().IntVarP(&timeLength, "time", "t", 1, "Length of a time-series for the input")
	createTrainTaskCmd.Flags().StringVarP(&clusterUuid, "cluster", "c", "", "Cluster UUID (mandatory)")
	createTrainTaskCmd.MarkFlagRequired("cluster")
	createTrainTaskCmd.MarkFlagRequired("input")
	createTrainTaskCmd.MarkFlagRequired("output")

	createTaskCmd.AddCommand(createRequestTaskCmd)
	createRequestTaskCmd.Flags().StringSliceVarP(&inputData, "input", "i", []string{}, "Cluster input, which are paris of '-i <HEXAGON_NAME>:<DATASET_UUID>' (mandatory)")
	createRequestTaskCmd.Flags().StringSliceVarP(&outputData, "result", "r", []string{}, "Cluster result, which are paris of '-o <HEXAGON_NAME>:<DATASET_NAME>' (mandatory)")
	createRequestTaskCmd.Flags().IntVarP(&timeLength, "time", "t", 1, "Length of a time-series for the input")
	createRequestTaskCmd.Flags().StringVarP(&clusterUuid, "cluster", "c", "", "Cluster UUID (mandatory)")
	createRequestTaskCmd.MarkFlagRequired("cluster")
	createRequestTaskCmd.MarkFlagRequired("input")
	createRequestTaskCmd.MarkFlagRequired("result")

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
