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
    "os"
    "hanamictl/common"
    "github.com/spf13/cobra"
    "github.com/kitsudaiki/Hanami"

    "golang.org/x/term"
)

var (
    userName    string
    password    string
    isAdmin     bool
)

var userHeader = []string{
    "id",
    "name",
    "is_admin",
    "projects",
    "creator_id",
    "created_at",
}

var createUserCmd = &cobra.Command {
    Use:   "create USER_ID",
    Short: "Create a new user.",
    Args:  cobra.ExactArgs(1),
    Run:   func(cmd *cobra.Command, args []string) {
        token := Login()
        address := os.Getenv("HANAMI_ADDRESS")
        var pw string
        var err error
        if len(password) == 0 {
            pw, err = getPassword()
            if err == nil {
                fmt.Printf("error: %s\n", err)
            }
        } else {
            pw = password
        }
        userId := args[0]

        content, err := hanami_sdk.CreateUser(address, token, userId, userName, pw, isAdmin)
        if err != nil {
            fmt.Println(err)
            os.Exit(1)
        }
        hanamictl_common.ParseSingle(content, userHeader)
    },
}

var getUserCmd = &cobra.Command {
    Use:   "get USER_ID",
    Short: "Get information of a specific user.",
    Args:  cobra.ExactArgs(1),
    Run:   func(cmd *cobra.Command, args []string) {
        token := Login()
        address := os.Getenv("HANAMI_ADDRESS")
        userId := args[0]
        content, err := hanami_sdk.GetUser(address, token, userId)
        if err != nil {
            fmt.Println(err)
            os.Exit(1)
        }
        hanamictl_common.ParseSingle(content, userHeader)
    },
}

var listUserCmd = &cobra.Command {
    Use:   "list",
    Short: "List all user.",
    Run:   func(cmd *cobra.Command, args []string) {
        token := Login()
        address := os.Getenv("HANAMI_ADDRESS")
        content, err := hanami_sdk.ListUser(address, token)
        if err != nil {
            fmt.Println(err)
            os.Exit(1)
        }
        hanamictl_common.ParseList(content, userHeader)
    },
}

var deleteUserCmd = &cobra.Command {
    Use:   "delete USER_ID",
    Short: "Delete a specific user from the backend.",
    Args:  cobra.ExactArgs(1),
    Run:   func(cmd *cobra.Command, args []string) {
        token := Login()
        address := os.Getenv("HANAMI_ADDRESS")
        userId := args[0]
        _, err := hanami_sdk.DeleteUser(address, token, userId)
        if err != nil {
            fmt.Println(err)
            os.Exit(1)
        }
        fmt.Printf("successfully deleted user '%v'\n", userId)
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

    passwordFlagText := "Password for the new user. " + 
        "If not given by this flag, the password will be automatically requested after entering the command. " +
        "This flag is quite unsave, because this way the password is visible in the command-line and " +
        "printed into the history. So this flag should be only used for automated testing, " +
        "but NEVER in a productive environment."
    userCmd.AddCommand(createUserCmd)
    createUserCmd.Flags().StringVarP(&userName, "name", "n", "", "User name (mandatory)")
    createUserCmd.Flags().StringVarP(&password, "password", "p", "", passwordFlagText)
    createUserCmd.Flags().BoolVar(&isAdmin, "is_admin", false, "Set user as admin (default: false)")
    createUserCmd.MarkFlagRequired("name")

    userCmd.AddCommand(getUserCmd)

    userCmd.AddCommand(listUserCmd)

    userCmd.AddCommand(deleteUserCmd)
}
