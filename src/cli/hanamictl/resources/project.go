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
    projectName    string
)

var createProjectCmd = &cobra.Command {
    Use:   "create PROJECT_ID",
    Short: "Create a new project.",
    Args:  cobra.ExactArgs(1),
    Run:   func(cmd *cobra.Command, args []string) {
        projectId := args[0]
        success, content := hanami_sdk.CreateProject(projectId, projectName)
        if success {
            hanamictl_common.ParseSingle(content)
        } else {
            fmt.Println(content)
        }
    },
}

var getProjectCmd = &cobra.Command {
    Use:   "get PROJECT_ID",
    Short: "Get information of a specific project.",
    Args:  cobra.ExactArgs(1),
    Run:   func(cmd *cobra.Command, args []string) {
        projectId := args[0]
        success, content := hanami_sdk.GetProject(projectId)
        if success {
            hanamictl_common.ParseSingle(content)
        } else {
            fmt.Println(content)
        }
    },
}

var listProjectCmd = &cobra.Command {
    Use:   "list",
    Short: "List all project.",
    Run:   func(cmd *cobra.Command, args []string) {
        success, content := hanami_sdk.ListProject()
        if success {
            hanamictl_common.ParseList(content)
        } else {
            fmt.Println(content)
        }
    },
}

var deleteProjectCmd = &cobra.Command {
    Use:   "delete PROJECT_ID",
    Short: "Delete a specific project from the backend.",
    Args:  cobra.ExactArgs(1),
    Run:   func(cmd *cobra.Command, args []string) {
        projectId := args[0]
        success, content := hanami_sdk.DeleteProject(projectId)
        if success {
            fmt.Println("successfully deleted project '%s'", projectId)
        } else {
            fmt.Println(content)
        }
    },
}


var projectCmd = &cobra.Command {
    Use:   "project",
    Short: "Manage project.",
}


func Init_Project_Commands(rootCmd *cobra.Command) {
    rootCmd.AddCommand(projectCmd)

    projectCmd.AddCommand(createProjectCmd)
    createProjectCmd.Flags().StringVarP(&projectName, "name", "n", "", "Project name (mandatory)")
    createProjectCmd.MarkFlagRequired("name")

    projectCmd.AddCommand(getProjectCmd)

    projectCmd.AddCommand(listProjectCmd)

    projectCmd.AddCommand(deleteProjectCmd)
}
