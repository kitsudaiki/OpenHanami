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
    inputFilePath string
    labelFilePath string
)

var createMnistDatasetCmd = &cobra.Command {
    Use:   "mnist DATASET_UUID",
    Short: "Upload new mnist dataset.",
    Args:  cobra.ExactArgs(1),
    Run:   func(cmd *cobra.Command, args []string) {
        token := Login()
        address := os.Getenv("HANAMI_ADDRESS")
        datasetName := args[0]
        success, content := hanami_sdk.UploadMnistFiles(address, token, datasetName, inputFilePath, labelFilePath)
        if success == false {
            fmt.Println(content)
        }

        success, content = hanami_sdk.GetDataset(address, token, content)
        if success {
            hanamictl_common.ParseSingle(content)
        } else {
            fmt.Println(content)
        }
    },
}

var createCsvDatasetCmd = &cobra.Command {
    Use:   "csv DATASET_UUID",
    Short: "Upload new csv dataset.",
    Args:  cobra.ExactArgs(1),
    Run:   func(cmd *cobra.Command, args []string) {
        token := Login()
        address := os.Getenv("HANAMI_ADDRESS")
        datasetName := args[0]
        success, content := hanami_sdk.UploadCsvFiles(address, token, datasetName, inputFilePath)
        if success == false {
            fmt.Println(content)
        }

        success, content = hanami_sdk.GetDataset(address, token, content)
        if success {
            hanamictl_common.ParseSingle(content)
        } else {
            fmt.Println(content)
        }
    },
}

var getDatasetCmd = &cobra.Command {
    Use:   "get DATASET_UUID",
    Short: "Get information of a specific dataset.",
    Args:  cobra.ExactArgs(1),
    Run:   func(cmd *cobra.Command, args []string) {
        token := Login()
        address := os.Getenv("HANAMI_ADDRESS")
        datasetUuid := args[0]
        success, content := hanami_sdk.GetDataset(address, token, datasetUuid)
        if success {
            hanamictl_common.ParseSingle(content)
        } else {
            fmt.Println(content)
        }
    },
}

var listDatasetCmd = &cobra.Command {
    Use:   "list",
    Short: "List all dataset.",
    Run:   func(cmd *cobra.Command, args []string) {
        token := Login()
        address := os.Getenv("HANAMI_ADDRESS")
        success, content := hanami_sdk.ListDataset(address, token)
        if success {
            hanamictl_common.ParseList(content)
        } else {
            fmt.Println(content)
        }
    },
}

var deleteDatasetCmd = &cobra.Command {
    Use:   "delete DATASET_UUID",
    Short: "Delete a specific dataset from the backend.",
    Args:  cobra.ExactArgs(1),
    Run:   func(cmd *cobra.Command, args []string) {
        token := Login()
        address := os.Getenv("HANAMI_ADDRESS")
        datasetUuid := args[0]
        success, content := hanami_sdk.DeleteDataset(address, token, datasetUuid)
        if success {
            fmt.Println("successfully deleted dataset '%s'", datasetUuid)
        } else {
            fmt.Println(content)
        }
    },
}


var datasetCmd = &cobra.Command {
    Use:   "dataset",
    Short: "Manage dataset.",
}

var createDatasetCmd = &cobra.Command {
    Use:   "create",
    Short: "Upload datasets.",
}

func Init_Dataset_Commands(rootCmd *cobra.Command) {
    rootCmd.AddCommand(datasetCmd)

    datasetCmd.AddCommand(createDatasetCmd)

    createDatasetCmd.AddCommand(createMnistDatasetCmd)
    createMnistDatasetCmd.Flags().StringVarP(&inputFilePath, "inputFilePath", "i", "", "Path to file with input-data (mandatory)")
    createMnistDatasetCmd.Flags().StringVarP(&labelFilePath, "labelFilePath", "l", "", "Path to file with label-data (mandatory)")
    createMnistDatasetCmd.MarkFlagRequired("inputFilePath")
    createMnistDatasetCmd.MarkFlagRequired("labelFilePath")

    createDatasetCmd.AddCommand(createCsvDatasetCmd)
    createCsvDatasetCmd.Flags().StringVarP(&inputFilePath, "inputFilePath", "i", "", "Path to file with input-data (mandatory)")
    createCsvDatasetCmd.MarkFlagRequired("inputFilePath")

    datasetCmd.AddCommand(getDatasetCmd)

    datasetCmd.AddCommand(listDatasetCmd)

    datasetCmd.AddCommand(deleteDatasetCmd)
}

