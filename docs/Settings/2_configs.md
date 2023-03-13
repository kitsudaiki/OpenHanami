# Configs

Each component has its own config-file, which is required for starting the serivce. 

## **Kyouko**

*Default-Path*: `/etc/kyouko/kyouko.conf`

**[DEFAULT]**

| Field | Type | Description |
| --- |  --- |  --- | 
| debug | bool | Set to true to enable debug-output |
| log_path | string | Directory-path, where the log-files should be written to. Directory must already exist. |
| address | string | Path to the unix-domain-socket file. File doesn't have to exist and will be created when starting the service. |
| database | string | Path to the sqlite database-file. If there is not already a database-file, it will automatically created and initialized when starting the service. |

**[misaki]**

| Field | Type | Description |
| --- |  --- |  --- | 
| address | string | Path to the unix-domain-socket file, where Misaki is listening. |

**[shiori]**

| Field | Type | Description |
| --- |  --- |  --- | 
| address | string | Path to the unix-domain-socket file, where Shiori is listening. |


!!! example

    ```
    [DEFAULT]
    #debug = true
    log_path = "/var/log/kyouko"
    address = "/tmp/hanami/kyouko.uds"
    database = "/etc/kyouko/kyouko_db"

    [misaki]
    address = "/tmp/hanami/misaki.uds"

    [shiori]
    address = "/tmp/hanami/shiori.uds"
    ```


## **Shiori**

*Default-Path*: `/etc/shiori/shiori.conf`

**[DEFAULT]**

| Field | Type | Description |
| --- |  --- |  --- | 
| debug | bool | Set to true to enable debug-output |
| log_path | string | Directory-path, where the log-files should be written to. Directory must already exist. |
| address | string | Path to the unix-domain-socket file. File doesn't have to exist and will be created when starting the service. |
| database | string | Path to the sqlite database-file. If there is not already a database-file, it will automatically created and initialized when starting the service. |


**[shiori]**

| Field | Type | Description |
| --- |  --- |  --- | 
| data_set_location | string | Path to directory, where the uploaded data-sets should be stored |
| cluster_snapshot_location | string | Path to directory, where the created snapshots coming from Kyouko should be stored |

**[misaki]**

| Field | Type | Description |
| --- |  --- |  --- | 
| address | string | Path to the unix-domain-socket file, where Misaki is listening. |

!!! example

    ```
    [DEFAULT]
    #debug = True
    log_path = "/var/log/shiori"
    address = "/tmp/hanami/shiori.uds"
    database = "/etc/shiori/shiori_db"

    [shiori]
    data_set_location = "/etc/shiori/files"
    cluster_snapshot_location = "/etc/shiori/files"

    [misaki]
    address = "/tmp/hanami/misaki.uds"
    ```

## **Misaki**

*Default-Path*: `/etc/misaki/misaki.conf`

**[DEFAULT]**

| Field | Type | Description |
| --- |  --- |  --- | 
| debug | bool | Set to true to enable debug-output |
| log_path | string | Directory-path, where the log-files should be written to. Directory must already exist. |
| address | string | Path to the unix-domain-socket file. File doesn't have to exist and will be created when starting the service. |
| database | string | Path to the sqlite database-file. If there is not already a database-file, it will automatically created and initialized when starting the service. |

**[misaki]**

| Field | Type | Description |
| --- |  --- |  --- | 
| token_key_path | string | Path to the file with the key for the JWT-Token-Creation and -Validation |
| policies | string | Path to policy-file |
| token_expire_time | int | Number of seconds until a new created JWT-token is expired |

**[azuki]**

| Field | Type | Description |
| --- |  --- |  --- | 
| address | string | Path to the unix-domain-socket file, where Azuki is listening. |

**[shiori]**

| Field | Type | Description |
| --- |  --- |  --- | 
| address | string | Path to the unix-domain-socket file, where Shiori is listening. |

**[kyouko]**

| Field | Type | Description |
| --- |  --- |  --- | 
| address | string | Path to the unix-domain-socket file, where Kyouko is listening. |

!!! example

    ```
    [DEFAULT]
    #debug = True
    log_path = "/var/log/misaki"
    address = "/tmp/hanami/misaki.uds"
    database = "/etc/misaki/misaki_db"

    [misaki]
    token_key_path = "/etc/misaki/token_key"
    policies = "/etc/misaki/policies"
    token_expire_time = {{ .Values.token.expire_time }}

    [azuki]
    address = "/tmp/azuki.uds"

    [shiori]
    address = "/tmp/hanami/shiori.uds"

    [kyouko]
    address = "/tmp/hanami/kyouko.uds"
    ```

## **Azuki**

*Default-Path*: `/etc/azuki/azuki.conf`


**[DEFAULT]**

| Field | Type | Description |
| --- |  --- |  --- | 
| debug | bool | Set to true to enable debug-output |
| log_path | string | Directory-path, where the log-files should be written to. Directory must already exist. |
| address | string | Path to the unix-domain-socket file. File doesn't have to exist and will be created when starting the service. |
| database | string | Path to the sqlite database-file. If there is not already a database-file, it will automatically created and initialized when starting the service. |

**[torii]**

| Field | Type | Description |
| --- |  --- |  --- | 
| address | string | Path to the unix-domain-socket file, where Torii is listening. |

**[misaki]**

| Field | Type | Description |
| --- |  --- |  --- | 
| address | string | Path to the unix-domain-socket file, where Misaki is listening. |

**[shiori]**

| Field | Type | Description |
| --- |  --- |  --- | 
| address | string | Path to the unix-domain-socket file, where Shiori is listening. |

**[kyouko]**

| Field | Type | Description |
| --- |  --- |  --- | 
| address | string | Path to the unix-domain-socket file, where Kyouko is listening. |

!!! example

    ```
    [DEFAULT]
    #debug = True
    log_path = "/var/log/azuki"
    sakura-file-locaion = "/etc/azuki/sakura-files"
    address = "/tmp/hanami/azuki.uds"
    database = "/etc/azuki/azuki_db"

    [torii]
    address = "/tmp/hanami/torii.uds"

    [misaki]
    address = "/tmp/hanami/misaki.uds"

    [shiori]
    address = "/tmp/hanami/shiori.uds"

    [kyouko]
    address = "/tmp/hanami/kyouko.uds"
    ```

## **Torii**

*Default-Path*: `/etc/torii/torii.conf`

**[DEFAULT]**

| Field | Type | Description |
| --- |  --- |  --- | 
| debug | bool | Set to true to enable debug-output |
| log_path | string | Directory-path, where the log-files should be written to. Directory must already exist. |
| address | string | Path to the unix-domain-socket file. File doesn't have to exist and will be created when starting the service. |

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

**[azuki]**

| Field | Type | Description |
| --- |  --- |  --- | 
| address | string | Path to the unix-domain-socket file, where Azuki is listening. |

**[misaki]**

| Field | Type | Description |
| --- |  --- |  --- | 
| address | string | Path to the unix-domain-socket file, where Misaki is listening. |

**[shiori]**

| Field | Type | Description |
| --- |  --- |  --- | 
| address | string | Path to the unix-domain-socket file, where Shiori is listening. |

**[kyouko]**

| Field | Type | Description |
| --- |  --- |  --- | 
| address | string | Path to the unix-domain-socket file, where Kyouko is listening. |

!!! example

    ```
    [DEFAULT]
    #debug = True
    log_path = "/var/log/torii"
    address = "/tmp/hanami/torii.uds"

    [http]
    enable = True
    ip = "0.0.0.0"
    certificate = "/etc/torii/cert.pem"
    key = "/etc/torii/key.pem"
    dashboard_files = "/etc/torii/Hanami-AI-Dashboard/src"
    port = 1337
    enable_dashboard = True

    [misaki]
    address = "/tmp/hanami/misaki.uds"

    [azuki]
    address = "/tmp/hanami/azuki.uds"

    [shiori]
    address = "/tmp/hanami/shiori.uds"

    [kyouko]
    address = "/tmp/hanami/kyouko.uds"
    ```


## **Tsugumi**

*Default-Path*: `/etc/tsugumi/tsugumi.conf`

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
| learn_inputs | string | Path to the mnist-file with inputs for training |
| learn_labels | string |Path to the mnist-file with labels for training |
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
    learn_inputs = "/home/neptune/Schreibtisch/mnist/train-images.idx3-ubyte"
    learn_labels = "/home/neptune/Schreibtisch/mnist/train-labels.idx1-ubyte"
    request_inputs = "/home/neptune/Schreibtisch/mnist/t10k-images.idx3-ubyte"
    request_labels = "/home/neptune/Schreibtisch/mnist/t10k-labels.idx1-ubyte"
    base_inputs = "/home/neptune/Schreibtisch/test.csv
    ```
