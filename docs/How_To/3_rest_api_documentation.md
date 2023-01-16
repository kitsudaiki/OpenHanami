# API documentation

!!! info

    This documentation was automatically created from the code to ensure consistency. Working alone on such a project with addtionally a high frequence in changing the API makes it nearly improssible to maintain this documentation manually. Beside this, it already saved me a huge amount of time, which I have available for the project itself instead.

!!! warning

    This documentation contains only the plain REST-API-endpoints. The workflow to upload files or interact with the Clusters over the websocket are not part of this documentation. [See instead](/How_To/4_websocket_workflow)

## misaki


### v1/auth

#### GET

Checks if a JWT-access-token of a user is valid or not and optional check if the user is allowed by its roles and the policy to access a specific endpoint.

**Request-Parameter**


`component`

**Description:** Requested component-name of the request. If this is not set, then only the token in itself will be validated.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | False |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

`endpoint`

**Description:** Requesed endpoint within the component.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | False |
| *Must match the regex* | [a-zA-Z][a-zA-Z_/0-9]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

`http_type`

**Description:** Type of the HTTP-request as enum (DELETE = 1, GET = 2, HEAD = 3, POST = 4, PUT = 5).

| attribute | value |
| --- | --- |
| *Type* | Int |
| *Required* | False |
| *Lower border of value* | 1 |
| *Upper border of value* | 5 |

`token`

**Description:** User specific JWT-access-token.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z_.\-0-9]* |

**Response-Parameter**


`id`

**Description:** ID of the user.

| attribute | value |
| --- | --- |
| *Type* | String |

`is_admin`

**Description:** Show if the user is an admin or not.

| attribute | value |
| --- | --- |
| *Type* | Bool |

`is_project_admin`

**Description:** True, if the user is admin within the selected project.

| attribute | value |
| --- | --- |
| *Type* | Bool |

`name`

**Description:** Name of the user.

| attribute | value |
| --- | --- |
| *Type* | String |

`project_id`

**Description:** Selected project of the user.

| attribute | value |
| --- | --- |
| *Type* | String |

`role`

**Description:** Role of the user within the project.

| attribute | value |
| --- | --- |
| *Type* | String |

### v1/bind_thread_to_core

#### POST

Bind threads of a specific thead-type-name to a specific core.

**Request-Parameter**


`core_ids`

**Description:** Core-ids to bind to.

| attribute | value |
| --- | --- |
| *Type* | Array |
| *Required* | True |

`thread_name`

**Description:** Thread-type-name of the threads, which should be bound to the core.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9\-]* |
| *Minimum string-length* | 3 |
| *Maximum string-length* | 256 |

**Response-Parameter**


### v1/documentation/api

#### GET

Generate a user-specific documentation for the API of the current component.

**Request-Parameter**


`type`

**Description:** Output-type of the document (pdf, rst, md).

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | False |
| *Default* | pdf |

**Response-Parameter**


`documentation`

**Description:** API-documentation as base64 converted string.

| attribute | value |
| --- | --- |
| *Type* | String |

### v1/documentation/api/rest

#### GET

Generate a documentation for the REST-API of all available components.

**Request-Parameter**


`type`

**Description:** Output-type of the document (pdf, rst, md).

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | False |
| *Default* | pdf |
| *Must match the regex* | ^(pdf|rst|md)$ |

**Response-Parameter**


`documentation`

**Description:** REST-API-documentation as base64 converted string.

| attribute | value |
| --- | --- |
| *Type* | String |

### v1/get_thread_mapping

#### GET

Collect all thread-names with its acutal mapped core-id's

**Request-Parameter**


**Response-Parameter**


`thread_map`

**Description:** Map with all thread-names and its core-id as json-string.

| attribute | value |
| --- | --- |
| *Type* | Map |

### v1/project

#### DELETE

Delete a specific user from the database.

**Request-Parameter**


`id`

**Description:** ID of the project.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

**Response-Parameter**


#### GET

Show information of a specific registered user.

**Request-Parameter**


`id`

**Description:** Id of the user.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

**Response-Parameter**


`creator_id`

**Description:** Id of the creator of the user.

| attribute | value |
| --- | --- |
| *Type* | String |

`id`

**Description:** ID of the new user.

| attribute | value |
| --- | --- |
| *Type* | String |

`name`

**Description:** Name of the new user.

| attribute | value |
| --- | --- |
| *Type* | String |

#### POST

Register a new project within Misaki.

**Request-Parameter**


`id`

**Description:** ID of the new project.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

`name`

**Description:** Name of the new project.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9 ]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

**Response-Parameter**


`creator_id`

**Description:** Id of the creator of the project.

| attribute | value |
| --- | --- |
| *Type* | String |

`id`

**Description:** ID of the new project.

| attribute | value |
| --- | --- |
| *Type* | String |

`name`

**Description:** Name of the new project.

| attribute | value |
| --- | --- |
| *Type* | String |

### v1/project/all

#### GET

Get information of all registered user as table.

**Request-Parameter**


**Response-Parameter**


`body`

**Description:** Array with all rows of the table, which array arrays too.

| attribute | value |
| --- | --- |
| *Type* | Array |

`header`

**Description:** Array with the namings all columns of the table.

| attribute | value |
| --- | --- |
| *Type* | Array |

### v1/token

#### POST

Create a JWT-access-token for a specific user.

**Request-Parameter**


`id`

**Description:** ID of the user.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9@]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

`password`

**Description:** Passphrase of the user, to verify the access.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Minimum string-length* | 8 |
| *Maximum string-length* | 4096 |

**Response-Parameter**


`id`

**Description:** ID of the user.

| attribute | value |
| --- | --- |
| *Type* | String |

`is_admin`

**Description:** Set this to true to register the new user as admin.

| attribute | value |
| --- | --- |
| *Type* | Bool |

`name`

**Description:** Name of the user.

| attribute | value |
| --- | --- |
| *Type* | String |

`token`

**Description:** New JWT-access-token for the user.

| attribute | value |
| --- | --- |
| *Type* | String |

#### PUT

Create a JWT-access-token for a specific user.

**Request-Parameter**


`project_id`

**Description:** ID of the project, which has to be used for the new token.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

**Response-Parameter**


`id`

**Description:** ID of the user.

| attribute | value |
| --- | --- |
| *Type* | String |

`is_admin`

**Description:** Set this to true to register the new user as admin.

| attribute | value |
| --- | --- |
| *Type* | Bool |

`name`

**Description:** Name of the user.

| attribute | value |
| --- | --- |
| *Type* | String |

`token`

**Description:** New JWT-access-token for the user.

| attribute | value |
| --- | --- |
| *Type* | String |

### v1/token/internal

#### POST

Create a JWT-access-token for a internal services, which can not be used from the outside.

**Request-Parameter**


`service_name`

**Description:** Name of the service.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

**Response-Parameter**


`token`

**Description:** New JWT-access-token for the service.

| attribute | value |
| --- | --- |
| *Type* | String |

### v1/user

#### DELETE

Delete a specific user from the database.

**Request-Parameter**


`id`

**Description:** ID of the user.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9@]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

**Response-Parameter**


#### GET

Show information of a specific user.

**Request-Parameter**


`id`

**Description:** Id of the user.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9@]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

**Response-Parameter**


`creator_id`

**Description:** Id of the creator of the user.

| attribute | value |
| --- | --- |
| *Type* | String |

`id`

**Description:** ID of the user.

| attribute | value |
| --- | --- |
| *Type* | String |

`is_admin`

**Description:** Set this to true to register the new user as admin.

| attribute | value |
| --- | --- |
| *Type* | Bool |

`name`

**Description:** Name of the user.

| attribute | value |
| --- | --- |
| *Type* | String |

`projects`

**Description:** Json-array with all assigned projects together with role and project-admin-status.

| attribute | value |
| --- | --- |
| *Type* | Array |

#### POST

Register a new user within Misaki.

**Request-Parameter**


`id`

**Description:** ID of the new user.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9@]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

`is_admin`

**Description:** Set this to 1 to register the new user as admin.

| attribute | value |
| --- | --- |
| *Type* | Bool |
| *Required* | False |
| *Default* | false |

`name`

**Description:** Name of the new user.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9 ]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

`password`

**Description:** Passphrase of the user.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Minimum string-length* | 8 |
| *Maximum string-length* | 4096 |

**Response-Parameter**


`creator_id`

**Description:** Id of the creator of the user.

| attribute | value |
| --- | --- |
| *Type* | String |

`id`

**Description:** ID of the new user.

| attribute | value |
| --- | --- |
| *Type* | String |

`is_admin`

**Description:** True, if user is an admin.

| attribute | value |
| --- | --- |
| *Type* | Bool |

`name`

**Description:** Name of the new user.

| attribute | value |
| --- | --- |
| *Type* | String |

`projects`

**Description:** Json-array with all assigned projects together with role and project-admin-status.

| attribute | value |
| --- | --- |
| *Type* | Array |

### v1/user/all

#### GET

Get information of all registered users.

**Request-Parameter**


**Response-Parameter**


`body`

**Description:** Array with all rows of the table, which array arrays too.

| attribute | value |
| --- | --- |
| *Type* | Array |

`header`

**Description:** Array with the namings all columns of the table.

| attribute | value |
| --- | --- |
| *Type* | Array |

### v1/user/project

#### DELETE

Remove a project from a specific user

**Request-Parameter**


`id`

**Description:** ID of the user.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9@]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

`project_id`

**Description:** ID of the project, which has to be removed from the user.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

**Response-Parameter**


`creator_id`

**Description:** Id of the creator of the user.

| attribute | value |
| --- | --- |
| *Type* | String |

`id`

**Description:** ID of the user.

| attribute | value |
| --- | --- |
| *Type* | String |

`is_admin`

**Description:** True, if user is an admin.

| attribute | value |
| --- | --- |
| *Type* | Bool |

`name`

**Description:** Name of the user.

| attribute | value |
| --- | --- |
| *Type* | String |

`projects`

**Description:** Json-array with all assigned projects together with role and project-admin-status.

| attribute | value |
| --- | --- |
| *Type* | Array |

#### GET

List all available projects of the user, who made the request.

**Request-Parameter**


`user_id`

**Description:** ID of the user.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | False |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9@]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

**Response-Parameter**


`projects`

**Description:** Json-array with all assigned projects together with role and project-admin-status.

| attribute | value |
| --- | --- |
| *Type* | Array |

#### POST

Add a project to a specific user.

**Request-Parameter**


`id`

**Description:** ID of the user.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9@]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

`is_project_admin`

**Description:** Set this to true, if the user should be an admin within the assigned project.

| attribute | value |
| --- | --- |
| *Type* | Bool |
| *Required* | False |
| *Default* | false |

`project_id`

**Description:** ID of the project, which has to be added to the user.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

`role`

**Description:** Role, which has to be assigned to the user within the project

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

**Response-Parameter**


`creator_id`

**Description:** Id of the creator of the user.

| attribute | value |
| --- | --- |
| *Type* | String |

`id`

**Description:** ID of the user.

| attribute | value |
| --- | --- |
| *Type* | String |

`is_admin`

**Description:** True, if user is an admin.

| attribute | value |
| --- | --- |
| *Type* | Bool |

`name`

**Description:** Name of the user.

| attribute | value |
| --- | --- |
| *Type* | String |

`projects`

**Description:** Json-array with all assigned projects together with role and project-admin-status.

| attribute | value |
| --- | --- |
| *Type* | Array |

## kyouko


### v1/bind_thread_to_core

#### POST

Bind threads of a specific thead-type-name to a specific core.

**Request-Parameter**


`core_ids`

**Description:** Core-ids to bind to.

| attribute | value |
| --- | --- |
| *Type* | Array |
| *Required* | True |

`thread_name`

**Description:** Thread-type-name of the threads, which should be bound to the core.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9\-]* |
| *Minimum string-length* | 3 |
| *Maximum string-length* | 256 |

**Response-Parameter**


### v1/cluster

#### DELETE

Delete a cluster.

**Request-Parameter**


`uuid`

**Description:** UUID of the cluster.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

**Response-Parameter**


#### GET

Show information of a specific cluster.

**Request-Parameter**


`uuid`

**Description:** uuid of the cluster.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

**Response-Parameter**


`name`

**Description:** Name of the cluster.

| attribute | value |
| --- | --- |
| *Type* | String |

`owner_id`

**Description:** ID of the user, who created the cluster.

| attribute | value |
| --- | --- |
| *Type* | String |

`project_id`

**Description:** ID of the project, where the cluster belongs to.

| attribute | value |
| --- | --- |
| *Type* | String |

`uuid`

**Description:** UUID of the cluster.

| attribute | value |
| --- | --- |
| *Type* | String |

`visibility`

**Description:** Visibility of the cluster (private, shared, public).

| attribute | value |
| --- | --- |
| *Type* | String |

#### POST

Create new cluster.

**Request-Parameter**


`name`

**Description:** Name for the new cluster.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9 ]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

`template`

**Description:** Cluster-template as base64-string.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | False |

**Response-Parameter**


`name`

**Description:** Name of the new created cluster.

| attribute | value |
| --- | --- |
| *Type* | String |

`owner_id`

**Description:** ID of the user, who created the new cluster.

| attribute | value |
| --- | --- |
| *Type* | String |

`project_id`

**Description:** ID of the project, where the new cluster belongs to.

| attribute | value |
| --- | --- |
| *Type* | String |

`uuid`

**Description:** UUID of the new created cluster.

| attribute | value |
| --- | --- |
| *Type* | String |

`visibility`

**Description:** Visibility of the new created cluster (private, shared, public).

| attribute | value |
| --- | --- |
| *Type* | String |

### v1/cluster/all

#### GET

List all visible clusters.

**Request-Parameter**


**Response-Parameter**


`body`

**Description:** Json-string with all information of all vilible clusters.

| attribute | value |
| --- | --- |
| *Type* | Array |

`header`

**Description:** Array with the names all columns of the table.

| attribute | value |
| --- | --- |
| *Type* | Array |

### v1/cluster/load

#### POST

Load a snapshot from shiori into an existing cluster and override the old data.

**Request-Parameter**


`cluster_uuid`

**Description:** UUID of the cluster, where the snapshot should be loaded into.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

`snapshot_uuid`

**Description:** UUID of the snapshot, which should be loaded from shiori into the cluster.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

**Response-Parameter**


`uuid`

**Description:** UUID of the load-task in the queue of the cluster.

| attribute | value |
| --- | --- |
| *Type* | String |

### v1/cluster/save

#### POST

Save a cluster.

**Request-Parameter**


`cluster_uuid`

**Description:** UUID of the cluster, which should be saved as new snapstho to shiori.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

`name`

**Description:** Name for task, which is place in the task-queue and for the new snapshot.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9 ]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

**Response-Parameter**


`name`

**Description:** Name of the new created task and of the snapshot, which should be created by the task.

| attribute | value |
| --- | --- |
| *Type* | String |

`uuid`

**Description:** UUID of the save-task in the queue of the cluster.

| attribute | value |
| --- | --- |
| *Type* | String |

### v1/cluster/set_mode

#### PUT

Set mode of the cluster.

**Request-Parameter**


`connection_uuid`

**Description:** UUID of the connection for input and output.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | False |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

`new_state`

**Description:** New desired state for the cluster.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | ^(TASK|DIRECT)$ |

`uuid`

**Description:** UUID of the cluster.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

**Response-Parameter**


`name`

**Description:** Name of the cluster.

| attribute | value |
| --- | --- |
| *Type* | String |

`new_state`

**Description:** New desired state for the cluster.

| attribute | value |
| --- | --- |
| *Type* | String |

`uuid`

**Description:** UUID of the cluster.

| attribute | value |
| --- | --- |
| *Type* | String |

### v1/documentation/api

#### GET

Generate a user-specific documentation for the API of the current component.

**Request-Parameter**


`type`

**Description:** Output-type of the document (pdf, rst, md).

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | False |
| *Default* | pdf |

**Response-Parameter**


`documentation`

**Description:** API-documentation as base64 converted string.

| attribute | value |
| --- | --- |
| *Type* | String |

### v1/get_thread_mapping

#### GET

Collect all thread-names with its acutal mapped core-id's

**Request-Parameter**


**Response-Parameter**


`thread_map`

**Description:** Map with all thread-names and its core-id as json-string.

| attribute | value |
| --- | --- |
| *Type* | Map |

### v1/task

#### DELETE

Delete a task or abort a task, if it is actually running.

**Request-Parameter**


`cluster_uuid`

**Description:** UUID of the cluster, which contains the task in its queue

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

`uuid`

**Description:** UUID of the task, which should be deleted

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

**Response-Parameter**


#### GET

Show information of a specific task.

**Request-Parameter**


`cluster_uuid`

**Description:** UUID of the cluster, which should process the request

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

`uuid`

**Description:** UUID of the cluster, which should process the request

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

**Response-Parameter**


`end_timestamp`

**Description:** Timestamp in UTC when the task was finished.

| attribute | value |
| --- | --- |
| *Type* | String |

`percentage_finished`

**Description:** Percentation of the progress between 0.0 and 1.0.

| attribute | value |
| --- | --- |
| *Type* | Float |

`queue_timestamp`

**Description:** Timestamp in UTC when the task entered the queued state, which is basicall the timestamp when the task was created

| attribute | value |
| --- | --- |
| *Type* | String |

`start_timestamp`

**Description:** Timestamp in UTC when the task entered the active state.

| attribute | value |
| --- | --- |
| *Type* | String |

`state`

**Description:** Actual state of the task (queued, active, aborted or finished).

| attribute | value |
| --- | --- |
| *Type* | String |

#### POST

Add new task to the task-queue of a cluster.

**Request-Parameter**


`cluster_uuid`

**Description:** UUID of the cluster, which should process the request

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

`data_set_uuid`

**Description:** UUID of the data-set with the input, which coming from shiori.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

`name`

**Description:** Name for the new task for better identification.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9 ]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

`type`

**Description:** UUID of the data-set with the input, which coming from shiori.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | ^(learn|request)$ |

**Response-Parameter**


`name`

**Description:** Name of the new created task.

| attribute | value |
| --- | --- |
| *Type* | String |

`uuid`

**Description:** UUID of the new created task.

| attribute | value |
| --- | --- |
| *Type* | String |

### v1/task/all

#### GET

List all visible tasks of a specific cluster.

**Request-Parameter**


`cluster_uuid`

**Description:** UUID of the cluster, whos tasks should be listed

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

**Response-Parameter**


`body`

**Description:** Array with all rows of the table, which array arrays too.

| attribute | value |
| --- | --- |
| *Type* | Array |

`header`

**Description:** Array with the namings all columns of the table.

| attribute | value |
| --- | --- |
| *Type* | Array |

### v1/template

#### DELETE

Delete a template from the database.

**Request-Parameter**


`uuid`

**Description:** UUID of the template.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

**Response-Parameter**


#### GET

Show a specific template.

**Request-Parameter**


`uuid`

**Description:** UUID of the template.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

**Response-Parameter**


`name`

**Description:** Name of the template.

| attribute | value |
| --- | --- |
| *Type* | String |

`owner_id`

**Description:** ID of the user, who created the template.

| attribute | value |
| --- | --- |
| *Type* | String |

`project_id`

**Description:** ID of the project, where the template belongs to.

| attribute | value |
| --- | --- |
| *Type* | String |

`template`

**Description:** The template itself.

| attribute | value |
| --- | --- |
| *Type* | String |

`uuid`

**Description:** UUID of the template.

| attribute | value |
| --- | --- |
| *Type* | String |

`visibility`

**Description:** Visibility of the template (private, shared, public).

| attribute | value |
| --- | --- |
| *Type* | String |

### v1/template/all

#### GET

List all visible templates.

**Request-Parameter**


**Response-Parameter**


`body`

**Description:** Array with all rows of the table, which array arrays too.

| attribute | value |
| --- | --- |
| *Type* | Array |

`header`

**Description:** Array with the namings all columns of the table.

| attribute | value |
| --- | --- |
| *Type* | Array |

### v1/template/upload

#### POST

Upload a new template and store it within the database.

**Request-Parameter**


`name`

**Description:** Name for the new template.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

`template`

**Description:** New template to upload as base64 string.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |

**Response-Parameter**


`name`

**Description:** Name of the new uploaded template.

| attribute | value |
| --- | --- |
| *Type* | String |

`owner_id`

**Description:** ID of the user, who created the new template.

| attribute | value |
| --- | --- |
| *Type* | String |

`project_id`

**Description:** ID of the project, where the new template belongs to.

| attribute | value |
| --- | --- |
| *Type* | String |

`uuid`

**Description:** UUID of the new uploaded template.

| attribute | value |
| --- | --- |
| *Type* | String |

`visibility`

**Description:** Visibility of the new created template (private, shared, public).

| attribute | value |
| --- | --- |
| *Type* | String |

## shiori


### v1/audit_log

#### GET

Get audit-log of a user.

**Request-Parameter**


`page`

**Description:** Page-number starting with 0 to access the logs. A page has up to 100 entries.

| attribute | value |
| --- | --- |
| *Type* | Int |
| *Required* | True |
| *Lower border of value* | 0 |
| *Upper border of value* | 1000000000 |

`user_id`

**Description:** ID of the user, whos entries are requested. Only an admin is allowed to set this values. Any other user get only its own log output based on the token-context.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | False |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9@]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

**Response-Parameter**


`body`

**Description:** Array with all rows of the table, which array arrays too.

| attribute | value |
| --- | --- |
| *Type* | Array |

`header`

**Description:** Array with the namings all columns of the table.

| attribute | value |
| --- | --- |
| *Type* | Array |

### v1/bind_thread_to_core

#### POST

Bind threads of a specific thead-type-name to a specific core.

**Request-Parameter**


`core_ids`

**Description:** Core-ids to bind to.

| attribute | value |
| --- | --- |
| *Type* | Array |
| *Required* | True |

`thread_name`

**Description:** Thread-type-name of the threads, which should be bound to the core.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9\-]* |
| *Minimum string-length* | 3 |
| *Maximum string-length* | 256 |

**Response-Parameter**


### v1/cluster_snapshot

#### DELETE

Delete a result-set from shiori.

**Request-Parameter**


`uuid`

**Description:** UUID of the cluster-snapshot to delete.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

**Response-Parameter**


#### GET

Get snapshot of a cluster.

**Request-Parameter**


`uuid`

**Description:** UUID of the original request-task, which placed the result in shiori.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

**Response-Parameter**


`header`

**Description:** Header-information of the snapshot-file.

| attribute | value |
| --- | --- |
| *Type* | Map |

`location`

**Description:** File path on local storage.

| attribute | value |
| --- | --- |
| *Type* | String |

`name`

**Description:** Name of the data-set.

| attribute | value |
| --- | --- |
| *Type* | String |

`uuid`

**Description:** UUID of the data-set.

| attribute | value |
| --- | --- |
| *Type* | String |

#### POST

Init new snapshot of a cluster.

**Request-Parameter**


`header`

**Description:** Header of the file with information of the splits.

| attribute | value |
| --- | --- |
| *Type* | Map |
| *Required* | True |

`input_data_size`

**Description:** Total size of the snapshot in number of bytes.

| attribute | value |
| --- | --- |
| *Type* | Int |
| *Required* | True |
| *Lower border of value* | 1 |
| *Upper border of value* | 10000000000 |

`name`

**Description:** Name of the new snapshot.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9 ]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

`project_id`

**Description:** ID of the project, where the snapshot belongs to.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

`user_id`

**Description:** ID of the user, who owns the snapshot.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9@]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

`uuid`

**Description:** UUID of the new snapshot.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

**Response-Parameter**


`name`

**Description:** Name of the new snapshot.

| attribute | value |
| --- | --- |
| *Type* | String |

`uuid`

**Description:** UUID of the new snapshot.

| attribute | value |
| --- | --- |
| *Type* | String |

`uuid_input_file`

**Description:** UUID to identify the file for data upload of the snapshot.

| attribute | value |
| --- | --- |
| *Type* | String |

#### PUT

Finish snapshot of a cluster.

**Request-Parameter**


`project_id`

**Description:** Name of the new set.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |

`user_id`

**Description:** ID of the user, who belongs to the snapshot.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

`uuid`

**Description:** Name of the new set.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

`uuid_input_file`

**Description:** UUID to identify the file for date upload of input-data.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

**Response-Parameter**


`uuid`

**Description:** UUID of the new set.

| attribute | value |
| --- | --- |
| *Type* | String |

### v1/cluster_snapshot/all

#### GET

List snapshots of all visible cluster.

**Request-Parameter**


**Response-Parameter**


`body`

**Description:** Array with all rows of the table, which array arrays too.

| attribute | value |
| --- | --- |
| *Type* | Array |

`header`

**Description:** Array with the namings all columns of the table.

| attribute | value |
| --- | --- |
| *Type* | Array |

### v1/csv/data_set

#### POST

Init new csv-file data-set.

**Request-Parameter**


`input_data_size`

**Description:** Total size of the input-data.

| attribute | value |
| --- | --- |
| *Type* | Int |
| *Required* | True |
| *Lower border of value* | 1 |
| *Upper border of value* | 10000000000 |

`name`

**Description:** Name of the new data-set.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9 ]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

**Response-Parameter**


`name`

**Description:** Name of the new data-set.

| attribute | value |
| --- | --- |
| *Type* | String |

`owner_id`

**Description:** ID of the user, who created the data-set.

| attribute | value |
| --- | --- |
| *Type* | String |

`project_id`

**Description:** ID of the project, where the data-set belongs to.

| attribute | value |
| --- | --- |
| *Type* | String |

`type`

**Description:** Type of the new set (csv)

| attribute | value |
| --- | --- |
| *Type* | String |

`uuid`

**Description:** UUID of the new data-set.

| attribute | value |
| --- | --- |
| *Type* | String |

`uuid_input_file`

**Description:** UUID to identify the file for date upload of input-data.

| attribute | value |
| --- | --- |
| *Type* | String |

`visibility`

**Description:** Visibility of the data-set (private, shared, public).

| attribute | value |
| --- | --- |
| *Type* | String |

#### PUT

Finalize uploaded data-set by checking completeness of the uploaded and convert into generic format.

**Request-Parameter**


`uuid`

**Description:** UUID of the new data-set.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

`uuid_input_file`

**Description:** UUID to identify the file for date upload of input-data.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

**Response-Parameter**


`uuid`

**Description:** UUID of the new data-set.

| attribute | value |
| --- | --- |
| *Type* | String |

### v1/data_set

#### DELETE

Delete a speific data-set.

**Request-Parameter**


`uuid`

**Description:** UUID of the data-set to delete.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

**Response-Parameter**


#### GET

Get information of a specific data-set.

**Request-Parameter**


`uuid`

**Description:** UUID of the data-set set to delete.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

**Response-Parameter**


`inputs`

**Description:** Number of inputs.

| attribute | value |
| --- | --- |
| *Type* | Int |

`lines`

**Description:** Number of lines.

| attribute | value |
| --- | --- |
| *Type* | Int |

`location`

**Description:** Local file-path of the data-set.

| attribute | value |
| --- | --- |
| *Type* | String |

`name`

**Description:** Name of the data-set.

| attribute | value |
| --- | --- |
| *Type* | String |

`outputs`

**Description:** Number of outputs.

| attribute | value |
| --- | --- |
| *Type* | Int |

`owner_id`

**Description:** ID of the user, who created the data-set.

| attribute | value |
| --- | --- |
| *Type* | String |

`project_id`

**Description:** ID of the project, where the data-set belongs to.

| attribute | value |
| --- | --- |
| *Type* | String |

`type`

**Description:** Type of the new set (csv or mnist)

| attribute | value |
| --- | --- |
| *Type* | String |

`uuid`

**Description:** UUID of the data-set.

| attribute | value |
| --- | --- |
| *Type* | String |

`visibility`

**Description:** Visibility of the data-set (private, shared, public).

| attribute | value |
| --- | --- |
| *Type* | String |

### v1/data_set/all

#### GET

List all visible data-sets.

**Request-Parameter**


**Response-Parameter**


`body`

**Description:** Array with all rows of the table, which array arrays too.

| attribute | value |
| --- | --- |
| *Type* | Array |

`header`

**Description:** Array with the namings all columns of the table.

| attribute | value |
| --- | --- |
| *Type* | Array |

### v1/data_set/check

#### POST

Compare a list of values with a data-set to check correctness.

**Request-Parameter**


`data_set_uuid`

**Description:** UUID of the data-set to compare to.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

`result_uuid`

**Description:** UUID of the data-set to compare to.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

**Response-Parameter**


`correctness`

**Description:** Correctness of the values compared to the data-set.

| attribute | value |
| --- | --- |
| *Type* | Float |

### v1/data_set/progress

#### GET

Get upload progress of a specific data-set.

**Request-Parameter**


`uuid`

**Description:** UUID of the dataset set to delete.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

**Response-Parameter**


`complete`

**Description:** True, if all temporary files for complete.

| attribute | value |
| --- | --- |
| *Type* | Bool |

`temp_files`

**Description:** Map with the uuids of the temporary files and it's upload progress

| attribute | value |
| --- | --- |
| *Type* | Map |

`uuid`

**Description:** UUID of the data-set.

| attribute | value |
| --- | --- |
| *Type* | String |

### v1/documentation/api

#### GET

Generate a user-specific documentation for the API of the current component.

**Request-Parameter**


`type`

**Description:** Output-type of the document (pdf, rst, md).

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | False |
| *Default* | pdf |

**Response-Parameter**


`documentation`

**Description:** API-documentation as base64 converted string.

| attribute | value |
| --- | --- |
| *Type* | String |

### v1/error_log

#### GET

Get error-log of a user. Only an admin is allowed to request the error-log.

**Request-Parameter**


`page`

**Description:** Page-number starting with 0 to access the logs. A page has up to 100 entries.

| attribute | value |
| --- | --- |
| *Type* | Int |
| *Required* | True |
| *Lower border of value* | 0 |
| *Upper border of value* | 1000000000 |

`user_id`

**Description:** ID of the user, whos entries are requested.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | False |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9@]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

**Response-Parameter**


`body`

**Description:** Array with all rows of the table, which array arrays too.

| attribute | value |
| --- | --- |
| *Type* | Array |

`header`

**Description:** Array with the namings all columns of the table.

| attribute | value |
| --- | --- |
| *Type* | Array |

### v1/get_thread_mapping

#### GET

Collect all thread-names with its acutal mapped core-id's

**Request-Parameter**


**Response-Parameter**


`thread_map`

**Description:** Map with all thread-names and its core-id as json-string.

| attribute | value |
| --- | --- |
| *Type* | Map |

### v1/mnist/data_set

#### POST

Init new mnist-file data-set.

**Request-Parameter**


`input_data_size`

**Description:** Total size of the input-data.

| attribute | value |
| --- | --- |
| *Type* | Int |
| *Required* | True |
| *Lower border of value* | 1 |
| *Upper border of value* | 10000000000 |

`label_data_size`

**Description:** Total size of the label-data.

| attribute | value |
| --- | --- |
| *Type* | Int |
| *Required* | True |
| *Lower border of value* | 1 |
| *Upper border of value* | 10000000000 |

`name`

**Description:** Name of the new set.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-zA-Z][a-zA-Z_0-9 ]* |
| *Minimum string-length* | 4 |
| *Maximum string-length* | 256 |

**Response-Parameter**


`name`

**Description:** Name of the new data-set.

| attribute | value |
| --- | --- |
| *Type* | String |

`owner_id`

**Description:** ID of the user, who created the data-set.

| attribute | value |
| --- | --- |
| *Type* | String |

`project_id`

**Description:** ID of the project, where the data-set belongs to.

| attribute | value |
| --- | --- |
| *Type* | String |

`type`

**Description:** Type of the new set (mnist)

| attribute | value |
| --- | --- |
| *Type* | String |

`uuid`

**Description:** UUID of the new data-set.

| attribute | value |
| --- | --- |
| *Type* | String |

`uuid_input_file`

**Description:** UUID to identify the file for date upload of input-data.

| attribute | value |
| --- | --- |
| *Type* | String |

`uuid_label_file`

**Description:** UUID to identify the file for date upload of label-data.

| attribute | value |
| --- | --- |
| *Type* | String |

`visibility`

**Description:** Visibility of the data-set (private, shared, public).

| attribute | value |
| --- | --- |
| *Type* | String |

#### PUT

Finalize uploaded data-set by checking completeness of the uploaded and convert into generic format.

**Request-Parameter**


`uuid`

**Description:** UUID of the new data-set.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

`uuid_input_file`

**Description:** UUID to identify the file for date upload of input-data.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

`uuid_label_file`

**Description:** UUID to identify the file for date upload of label-data.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

**Response-Parameter**


`uuid`

**Description:** UUID of the new data-set.

| attribute | value |
| --- | --- |
| *Type* | String |

### v1/request_result

#### DELETE

Delete a request-result from shiori.

**Request-Parameter**


`uuid`

**Description:** UUID of the original request-task, which placed the result in shiori.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

**Response-Parameter**


#### GET

Get a specific request-result

**Request-Parameter**


`uuid`

**Description:** UUID of the original request-task, which placed the result in shiori.

| attribute | value |
| --- | --- |
| *Type* | String |
| *Required* | True |
| *Must match the regex* | [a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12} |

**Response-Parameter**


`data`

**Description:** Result of the request-task as json-array.

| attribute | value |
| --- | --- |
| *Type* | Array |

`name`

**Description:** Name of the request-result.

| attribute | value |
| --- | --- |
| *Type* | String |

`owner_id`

**Description:** ID of the user, who created the request-result.

| attribute | value |
| --- | --- |
| *Type* | String |

`project_id`

**Description:** ID of the project, where the request-result belongs to.

| attribute | value |
| --- | --- |
| *Type* | String |

`uuid`

**Description:** UUID of the request-result.

| attribute | value |
| --- | --- |
| *Type* | String |

`visibility`

**Description:** Visibility of the request-result (private, shared, public).

| attribute | value |
| --- | --- |
| *Type* | String |

### v1/request_result/all

#### GET

List all visilbe request-results.

**Request-Parameter**


**Response-Parameter**


`body`

**Description:** Array with all rows of the table, which array arrays too.

| attribute | value |
| --- | --- |
| *Type* | Array |

`header`

**Description:** Array with the namings all columns of the table.

| attribute | value |
| --- | --- |
| *Type* | Array |
