# Configs

## **Hanami**

*Default-Path*: `/etc/hanami/hanami.conf`

**[DEFAULT]**

| Field | Type | Description |
| --- |  --- |  --- | 
| debug | bool | Set to true to enable debug-output |
| log_path | string | Directory-path, where the log-files should be written to. Directory must already exist. |
| address | string | Path to the unix-domain-socket file. File doesn't have to exist and will be created when starting the service. |

**[storage]**

| Field | Type | Description |
| --- |  --- |  --- | 
| data_set_location | string | Path to directory, where the uploaded data-sets should be stored |
| checkpoint_location | string | Path to directory, where the created checkpoints coming from Kyouko should be stored |

**[auth]**

| Field | Type | Description |
| --- |  --- |  --- | 
| token_key_path | string | Path to the file with the key for the JWT-Token-Creation and -Validation |
| policies | string | Path to policy-file |
| token_expire_time | int | Number of seconds until a new created JWT-token is expired |

**[http]**

| Field | Type | Description |
| --- |  --- |  --- | 
| enable | bool | Set true to enable HTTP-server (DEPRECATED) |
| ip | string | IP-address where the server should listen |
| port | int | Port-number where the server should listen |
| certificate | string | Path to the SSL-Cert-file |
| key | string | Path to the SSL-Key-file |
| dashboard_files | string | Path to directory, which contains the source-code of the dashboard | 
| enable_dashboard | bool | True to provide Dashboard |

!!! example

    ```
    [DEFAULT]
    debug = True
    log_path = "/var/log"
    database = "/etc/hanami/hanami_db"
    socket_path = "/tmp/hanami"

    [storage]
    data_set_location = "/etc/hanami/train_data"
    checkpoint_location = "/etc/hanami/checkpoints"

    [auth]
    policies = "/etc/hanami/policies"
    token_key_path = "/etc/hanami/token_key"
    token_expire_time = 3600

    [http]
    enable = True
    ip = "0.0.0.0"
    certificate = "/etc/torii/cert.pem"
    key = "/etc/torii/key.pem"
    dashboard_files = "/etc/hanami/frontend/Hanami-AI-Dashboard/src"
    port = 1337
    enable_dashboard = True

    [azuki]
    enable = False
    ```


## **SDK_API_Testing**

*Default-Path*: `/etc/hanami/hanami_testing.conf`

**[DEFAULT]**

| Field | Type | Description |
| --- |  --- |  --- | 
| debug | bool | Set to true to enable debug-output |

**[connection]**

| Field | Type | Description |
| --- |  --- |  --- | 
| host | string | IP of the Torii-server |
| port | int | Port on which the Torii is listening |
| test_user | string | user-ID, which should be used for the tests |
| test_pw | string | Passphrase of the test_user |


**[test_data]**

| Field | Type | Description |
| --- |  --- |  --- | 
| type | string | Type of the used test (mnist or csv) |
| train_inputs | string | Path to the mnist-file with inputs for training |
| train_labels | string | Path to the mnist-file with labels for training |
| request_inputs | string | Path to the mnist-file with inputs for testing |
| request_labels | string | Path to the mnist-file with labels for testing |
| base_inputs | string | Path to a CSV-file with test-data |

!!! example

    ```
    [DEFAULT]
    debug = true

    [connection]
    host = "127.0.0.1"
    port = 1337
    test_user = "test_user"
    test_pw = "asdfasdf"

    [test_data]
    type = "mnist"
    train_inputs = "/home/neptune/Schreibtisch/mnist/train-images.idx3-ubyte"
    train_labels = "/home/neptune/Schreibtisch/mnist/train-labels.idx1-ubyte"
    request_inputs = "/home/neptune/Schreibtisch/mnist/t10k-images.idx3-ubyte"
    request_labels = "/home/neptune/Schreibtisch/mnist/t10k-labels.idx1-ubyte"
    base_inputs = "/home/neptune/Schreibtisch/test.csv
    ```
