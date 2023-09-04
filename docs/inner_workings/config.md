# Configs of Hanami

!!! info

    This documeation was generated from the source-code to provide a maximum of consistency.
   
## DEFAULT

| Item | Description |
| --- | --- |
| database| **Description**: Path to the sqlite3 database-file for all local sql-tables of hanami.<br>**Required**: FALSE<br>**Default**: /etc/hanami/hanami_db<br> |
| debug| **Description**: Flag to enable debug-output in logging.<br>**Required**: FALSE<br>**Default**: false<br> |
| log_path| **Description**: Path to the directory, where the log-files should be written into.<br>**Required**: FALSE<br>**Default**: /var/log<br> |
| use_cuda| **Description**: Use very experimental CUDA processing.<br>**Required**: FALSE<br>**Default**: false<br> |

## auth

| Item | Description |
| --- | --- |
| policies| **Description**: Local path to the file with the endpoint-policies.<br>**Required**: FALSE<br>**Default**: /etc/hanami/policies<br> |
| token_expire_time| **Description**: Number of seconds, until a jwt-token expired.<br>**Required**: FALSE<br>**Default**: 3600<br> |
| token_key_path| **Description**: Local path to the file with the key for signing and validating the jwt-token.<br>**Required**: TRUE<br> |

## http

| Item | Description |
| --- | --- |
| certificate| **Description**: Local path to the file with the certificate for the https-connection.<br>**Required**: TRUE<br> |
| dashboard_files| **Description**: Local path to the directory, which contains the files of the dashboard.<br>**Required**: TRUE<br> |
| enable| **Description**: Flag to enable the http-endpoint.<br>**Required**: FALSE<br>**Default**: false<br> |
| enable_dashboard| **Description**: Flag to enable the dashboard.<br>**Required**: FALSE<br>**Default**: false<br> |
| ip| **Description**: IP-address, where the http-server should listen.<br>**Required**: FALSE<br>**Default**: 0.0.0.0<br> |
| key| **Description**: Local path to the file with the key for the https-connection.<br>**Required**: TRUE<br> |
| number_of_threads| **Description**: Number of threads in the thread-pool for processing http-requests.<br>**Required**: FALSE<br>**Default**: 4<br> |
| port| **Description**: Port, where the http-server should listen.<br>**Required**: TRUE<br> |

## storage

| Item | Description |
| --- | --- |
| checkpoint_location| **Description**: Local storage location, where all uploaded data-set should be written into.<br>**Required**: FALSE<br>**Default**: /etc/hanami/cluster_snapshots<br> |
| data_set_location| **Description**: Local storage location, where all uploaded data-set should be written into.<br>**Required**: FALSE<br>**Default**: /etc/hanami/train_data<br> |

