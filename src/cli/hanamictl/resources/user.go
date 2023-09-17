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
    "strings"
    "syscall"
    "errors"
    
    "hanamictl/common"
    "github.com/spf13/cobra"
    "github.com/kitsudaiki/Hanami"

    "golang.org/x/term"
)

var (
    userName    string
    isAdmin     bool
)

var createUserCmd = &cobra.Command {
    Use:   "create USER_ID",
    Short: "Create a new user.",
    Args:  cobra.ExactArgs(1),
    Run:   func(cmd *cobra.Command, args []string) {
        pw, err := getPassword()
        userId := args[0]

        if err == nil {
            success, content := hanami_sdk.CreateUser(userId, userName, pw, isAdmin)
            if success {
                hanamictl_common.ParseSingle(content)
            } else {
                fmt.Println(content)
            }
        } else {
            fmt.Printf("error: %s\n", err)
        }
    },
}

var getUserCmd = &cobra.Command {
    Use:   "get USER_ID",
    Short: "Get information of a specific user.",
    Args:  cobra.ExactArgs(1),
    Run:   func(cmd *cobra.Command, args []string) {
        userId := args[0]
        success, content := hanami_sdk.GetUser(userId)
        if success {
            hanamictl_common.ParseSingle(content)
        } else {
            fmt.Println(content)
        }
    },
}

var listUserCmd = &cobra.Command {
    Use:   "list",
    Short: "List all user.",
    Run:   func(cmd *cobra.Command, args []string) {
        success, content := hanami_sdk.ListUser()
        if success {
            hanamictl_common.ParseList(content)
        } else {
            fmt.Println(content)
        }
    },
}

var deleteUserCmd = &cobra.Command {
    Use:   "delete USER_ID",
    Short: "Delete a specific user from the backend.",
    Args:  cobra.ExactArgs(1),
    Run:   func(cmd *cobra.Command, args []string) {
        userId := args[0]
        success, content := hanami_sdk.DeleteUser(userId)
        if success {
            fmt.Println("successfully deleted user '%s'", userId)
        } else {
            fmt.Println(content)
        }
    },
}


var userCmd = &cobra.Command {
    Use:   "user",
    Short: "Manage user.",
}


func getPassword() (string, error) {
    fmt.Print("Enter Password: ")
    bytePassword1, err := term.ReadPassword(int(syscall.Stdin))
    if err != nil {
        return "", err
    }

    password1 := strings.TrimSpace(string(bytePassword1))

    fmt.Print("\n")
    fmt.Print("Enter Password again: ")
    bytePassword2, err := term.ReadPassword(int(syscall.Stdin))
    if err != nil {
        return "", err
    }

    password2 := strings.TrimSpace(string(bytePassword2))
    
    fmt.Print("\n")
    if password1 != password2 {
        return "", errors.New("Mismatch between the two entered passwords")
    }

    return password1, nil
}


func Init_User_Commands(rootCmd *cobra.Command) {
    rootCmd.AddCommand(userCmd)

    userCmd.AddCommand(createUserCmd)
    createUserCmd.Flags().StringVarP(&userName, "name", "n", "", "User name (mandatory)")
    createUserCmd.Flags().BoolVar(&isAdmin, "is_admin", false, "Set user as admin (default: false)")
    createUserCmd.MarkFlagRequired("name")

    userCmd.AddCommand(getUserCmd)

    userCmd.AddCommand(listUserCmd)

    userCmd.AddCommand(deleteUserCmd)
}
