/**
 * @author      Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright   Apache License Version 2.0
 *
 *      Copyright 2022-2026 Tobias Anker <tobias.anker@kitsunemimi.moe>
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

package ainari_resources

import (
	"fmt"
	ainarictl_common "ainarictl/common"
	"os"

	ainari_sdk "github.com/kitsudaiki/ainari"
	"github.com/spf13/cobra"
)

var (
	templatePath   string
	checkpointName string
	instanceMode    string
)

var createInstanceCmd = &cobra.Command{
	Use:   "create -t TEMPLATE_PATH NAME",
	Short: "Create a new instance.",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		context, err := Login()
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
		instanceName := args[0]
		templateContent, err := os.ReadFile(templatePath)
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
		content, err := ainari_sdk.CreateInstance(context, instanceName, string(templateContent))
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
		ainarictl_common.PrintSingle(content)
	},
}

var getInstanceCmd = &cobra.Command{
	Use:   "get CLUSTER_UUID",
	Short: "Get information of a specific instance.",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		context, err := Login()
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
		instanceUuid := args[0]
		content, err := ainari_sdk.GetInstance(context, instanceUuid)
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
		ainarictl_common.PrintSingle(content)
	},
}

var listInstanceCmd = &cobra.Command{
	Use:   "list",
	Short: "List all instance.",
	Run: func(cmd *cobra.Command, args []string) {
		context, err := Login()
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
		content, err := ainari_sdk.ListInstance(context)
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
		ainarictl_common.PrintList(content["instances"].([]interface{}))
	},
}

var deleteInstanceCmd = &cobra.Command{
	Use:   "delete CLUSTER_UUID",
	Short: "Delete a specific instance from the backend.",
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		context, err := Login()
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
		instanceUuid := args[0]
		_, err = ainari_sdk.DeleteInstance(context, instanceUuid)
		if err != nil {
			fmt.Println(err)
			os.Exit(1)
		}
		fmt.Printf("successfully deleted instance '%v'\n", instanceUuid)
	},
}

var instanceCmd = &cobra.Command{
	Use:   "instance",
	Short: "Manage instance.",
}

func Init_Instance_Commands(rootCmd *cobra.Command) {
	rootCmd.AddCommand(instanceCmd)

	instanceCmd.AddCommand(createInstanceCmd)
	createInstanceCmd.Flags().StringVarP(&templatePath, "template", "t", "", "Instance Template (mandatory)")
	createInstanceCmd.MarkFlagRequired("template")

	instanceCmd.AddCommand(getInstanceCmd)

	instanceCmd.AddCommand(listInstanceCmd)

	instanceCmd.AddCommand(deleteInstanceCmd)
}
