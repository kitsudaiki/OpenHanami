# CLI-client

!!! warning

    This is only a first quick-and-dirty PoC.

## Preparation

- Switch into the go-SDK directory

    ```
    cd src/sdk/go/hanami_sdk
    ```

- Build the protobuf-message for go

    ```
    protoc --go_out=. --proto_path ../../../libraries/hanami_messages/protobuffers hanami_messages.proto3
    ```

- Replace the package-name within the created protobuf-message

    ```
    sed -i 's/hanami_messages/hanami_sdk/g' hanami_messages.proto3.pb.go
    ```

- Switch into the directory of the client

    ```
    cd ../../../cli/hanamictl/
    ```

- Build `hanamictl`-binary:

    ```
    go build .
    ```

- load environment-variables

    ```
    export HANAMI_ADDRESS=http://127.0.0.1:11418
    export HANAMI_USER=asdf
    export HANAMI_PW=asdfasdf
    ```

    !!! info

        Port `11418` is the default-port of Hanami.

## Usage example

- Help-output

    ```
    ./hanamictl user create --help
    Create a new user.

    Usage:
      hanamictl user create USER_ID [flags]

    Flags:
      -h, --help          help for create
          --is_admin      Set user as admin (default: false)
      -n, --name string   User name (mandatory)
    ```

- Create-command

    ```
    ./hanamictl user create --name "CLI User" cli_user
    Enter Password: 
    Enter Password again: 
    +------------+----------+
    | NAME       | CLI User |
    +------------+----------+
    | PROJECTS   | []       |
    +------------+----------+
    | CREATOR_ID | asdf     |
    +------------+----------+
    | ID         | cli_user |
    +------------+----------+
    | IS_ADMIN   | false    |
    +------------+----------+
    ```

- List-command

    ```
    ./hanamictl user list        
                         
    +----------+----------+------------+----------+----------+
    |    ID    |   NAME   | CREATOR ID | PROJECTS | IS ADMIN |
    +----------+----------+------------+----------+----------+
    | asdf     | asdf     | MISAKI     | []       | true     |
    +----------+----------+------------+----------+----------+
    | cli_user | CLI User | asdf       | []       | false    |
    +----------+----------+------------+----------+----------+
    ```
