# Database Tables

!!! info

    At the moment all components, which are using a database to hold some information, are using `SQLite`.

## **Kyouko**

### *clusters*

| Column-Name | Type | Max-Length | is primary key | Description | 
| --- | --- | --- | --- | --- |
| uuid | varchar | 36 | x | UUID of the resource |
| projectId | varchar | 128 |   | ID of the project, where the resource belongs to |
| owner_id | varchar | 128 |   | ID of the user, who owns the resource |
| visibility | varchar | 10 |   | Visibility of the resource |
| name | varchar | 256 |   | Readable name of the resource |

### *templates*

| Column-Name | Type | Max-Length | is primary key | Description | 
| --- | --- | --- | --- | --- |
| uuid | varchar | 36 | x | UUID of the resource |
| projectId | varchar | 128 |   | ID of the project, where the resource belongs to |
| owner_id | varchar | 128 |   | ID of the user, who owns the resource |
| visibility | varchar | 10 |   | Visibility of the resource |
| name | varchar | 256 |   | Readable name of the resource |
| data | text |  |   |  |

## **Shiori**

### *cluster_snapshot*

| Column-Name | Type | Max-Length | is primary key | Description | 
| --- | --- | --- | --- | --- |
| uuid | varchar | 36 | x | UUID of the resource |
| projectId | varchar | 128 |   | ID of the project, where the resource belongs to |
| owner_id | varchar | 128 |   | ID of the user, who owns the resource |
| visibility | varchar | 10 |   | Visibility of the resource |
| name | varchar | 256 |   | Readable name of the resource |
| header | text |  |  | Json-formated header with additional information |
| location | text |  |  | ath of the local file-location of the file |
| temp_files | text |  |  | Json-formated map with names and progress of the temporary files while uploading |

### *data_set*

| Column-Name | Type | Max-Length | is primary key | Description | 
| --- | --- | --- | --- | --- |
| uuid | varchar | 36 | x | UUID of the resource |
| projectId | varchar | 128 |   | ID of the project, where the resource belongs to |
| owner_id | varchar | 128 |   | ID of the user, who owns the resource |
| visibility | varchar | 10 |   | Visibility of the resource |
| name | varchar | 256 |   | Readable name of the resource |
| type | varchar | 64 |  | Type of the data-set ("mnist" or "csv") |
| location | text |  |  | Path of the local file-location of the file |
| temp_files | text |  |  | Json-formated map with names and progress of the temporary files while uploading |

### *request_result*

| Column-Name | Type | Max-Length | is primary key | Description | 
| --- | --- | --- | --- | --- |
| uuid | varchar | 36 | x | UUID of the resource |
| projectId | varchar | 128 |   | ID of the project, where the resource belongs to |
| owner_id | varchar | 128 |   | ID of the user, who owns the resource |
| visibility | varchar | 10 |   | Visibility of the resource |
| name | varchar | 256 |   | Name of the task, which produced the results |
| data | text |  |   | Json-formated string with the results of all requests |

### *audit_log*

| Column-Name | Type | Max-Length | is primary key | Description | 
| --- | --- | --- | --- | --- |
| timestamp | varchar | 128 |   | Timestamp of the request |
| user_id | varchar | 256 |   | ID of the user, who made the request |
| component | varchar | 128 |   | Name of the component, which was the target of the request |
| endpoint | varchar | 1024 |   | Endpoint-path, which was requested |
| request_type | varchar | 16 |   | HTTP-Request-Type (GET, POST, PUT, DELETE) |

### *error_log*

| Column-Name | Type | Max-Length | is primary key | Description | 
| --- | --- | --- | --- | --- |
| timestamp | varchar | 128 |   | Timestamp, when the error occurred |
| user_id | varchar | 256 |   | ID of the user, who had the error |
| component | varchar | 128 |   | Component in which the error appeared |
| context | text | |   | Context-object as json-string with user-information |
| input_values | text | |   | Input-Values of the request, which produced the error |
| message | text | |   | Error-message of the error. |

## **Misaki**

### *users*

| Column-Name | Type | Max-Length | is primary key | Description | 
| --- | --- | --- | --- | --- |
| id | varchar | 36 | x | Unique identifier for the user |
| name | varchar | 36 |   | Name of the user |
| creator_id | varchar | 128 |   | ID of the user, who added the user |
| projects | text | |   | Json-array with all assigned projects together with role and project-admin-status. |
| is_admin | bool | |   | Shows if the user is a global admin, or not |
| pw_hash | varchar | 64 |   | Sha256-Hash of the passphrase and salt-string |
| salt | varchar | 64 |   | Salt-string for the pw_hash |


### *projects*

| Column-Name | Type | Max-Length | is primary key | Description | 
| --- | --- | --- | --- | --- |
| id | varchar | 36 | x | Unique identifier for the project |
| name | varchar | 36 |   | Name of the project |
| creator_id | varchar | 128 |   | ID of the user, who created the project |


