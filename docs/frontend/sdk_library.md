# SDK-library

The SDK-library privides functions to interact with the API of the backend. At the moment 2 versions are provided:

- `Python`
- (`Go`)
- (`Javascript`)

!!! info

    This documentation is not automatically generated from the source-code. So if you find outdated or broken function in this documentation, then please create an issue on github or fix this by yourself and create a pull-request.

## Installation


=== "Python"

    ```bash
    # clone repository
    git clone https://github.com/kitsudaiki/Hanami.git

    # create python-env (optional)
    python3 -m venv hanami_sdk_env
    source hanami_sdk_env/bin/activate

    # install sdk
    cd Hanami/src/sdk/python/hanami_sdk
    pip3 install -U .
    ```

## Exceptions

Each of the used HTTP-error codes results in a different exception. For the available error-code / exceptions of each of the endpoints, look into the [REST-API documenation](https://docs.hanami-ai.com/api/rest_api_documentation/)

=== "Python"

    ```python
    from hanami_sdk import hanami_exceptions

    try:
        (command)
    except hanami_exceptions.NotFoundException as e:
        print(e)
    except hanami_exceptions.UnauthorizedException as e:
        print(e)
    except hanami_exceptions.BadRequestException as e:
        print(e)
    except hanami_exceptions.ConflictException as e:
        print(e)
    except hanami_exceptions.InternalServerErrorException as e:
        print("internal error")
    ```

    !!! info

        The `InternalServerErrorException` doesn't contain a message. If this exception appears, you have have to look into the logs on the server.

## For insecure connections

In case the server use self-signed certificates for its https-connection, the ssl verification can be disabled. Each functions has a paramater `verify_connection`, wich is per default `True`. This validation can be disabled by adding `,verify_connection=False` to the end of a function-call.

## Request Token

For each of the following actions, the user must request an access-token at the beginning. This token is a jwt-token with basic information of the user. The token is only valid for a certain amount of time until it expires, based on the configuration of the server.

=== "Python"

    ```python
    from hanami_sdk import hanami_token

    address = "http://127.0.0.1:11418"
    test_user = "asdf"
    test_pw = "asdfasdf"

    token = hanami_token.request_token(address, test_user, test_pw)

    ```


## Project

Non-admin user need to be assigned to a project for logical separation.

!!! info

    These endpoints have a hard-coded requirement, that only admins are allowed to manage projects.

### Create Project

Create new empty project.

=== "Python"

    ```python
    from hanami_sdk import project

    address = "http://127.0.0.1:11418"
    project_id = "test_project"
    project_name = "Test Project"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    result = project.create_project(token, address, projet_id, project_name)

    # example-content of result:
    #
    # {
    #     "creator_id": "asdf",
    #     "id": "test_project",
    #     "name": "Test Project"
    # }
    ```

### Get Project

Get information about a project.

=== "Python"

    ```python
    from hanami_sdk import project

    address = "http://127.0.0.1:11418"
    project_id = "test_project"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    result = project.get_project(token, address, projet_id)

    # example-content of result:
    #
    # {
    #     "creator_id": "asdf",
    #     "id": "test_project",
    #     "name": "Test Project"
    # }
    ```

### List Project

List all projects.

=== "Python"

    ```python
    from hanami_sdk import project

    address = "http://127.0.0.1:11418"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    result = project.list_projects(token, address)

    # example-content of result:
    #
    # {
    #     "body": [
    #         [
    #             "test_project",
    #             "Test Project",
    #             "asdf"
    #         ]
    #     ],
    #     "header": [
    #         "id",
    #         "name",
    #         "creator_id"
    #     ]
    # }
    ```

### Delete Project

Delete a project.

!!! warning

    At the moment there is no check, if there still exist resources within this project.

=== "Python"

    ```python
    from hanami_sdk import project

    address = "http://127.0.0.1:11418"
    project_id = "test_project"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    project.delete_project(token, address, projet_id)
    ```


## User

!!! info

    These endpoints have a hard-coded requirement, that only admins are allowed to manage user.

### Create User

Create a new user.

If the `is_admin` is set to true, the user becomes a global admin. 

=== "Python"

    ```python
    from hanami_sdk import user

    address = "http://127.0.0.1:11418"
    new_user = "new_user"
    new_id = "new_user"
    new_pw = "asdfasdf"
    is_admin = True

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    result = user.create_user(token, address, new_id, new_user, new_pw, is_admin)

    # example-content of result:
    #
    # {
    #     "creator_id": "asdf",
    #     "id": "new_user",
    #     "is_admin": true,
    #     "name": "new_user",
    #     "projects": []
    # }
    ```

### Get User

Get information about a specific user.

=== "Python"

    ```python
    from hanami_sdk import user

    address = "http://127.0.0.1:11418"
    user_id = "new_user"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    result = user.get_user(token, address, user_id)

    # example-content of result:
    #
    # {
    #     "creator_id": "asdf",
    #     "id": "tsugumi",
    #     "is_admin": true,
    #     "name": "Tsugumi",
    #     "projects": []
    # }
    ```

### List User

List all user.

=== "Python"

    ```python
    from hanami_sdk import user

    address = "http://127.0.0.1:11418"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    result = user.list_users(token, address)

    # example-content of result:
    #
    # {
    #     "header": [
    #         "id",
    #         "name",
    #         "creator_id",
    #         "projects",
    #         "is_admin"
    #     ],
    #     "body": [
    #         [
    #             "asdf",
    #             "asdf",
    #             "MISAKI",
    #             [],
    #             true
    #         ],
    #         [
    #             "new_user",
    #             "new_user",
    #             "asdf",
    #             [],
    #             true
    #         ]
    #     ]
    # }
    ```

### Delete User

Delete a user from the backend.

!!! info

    A user can not be deleted by himself.

=== "Python"

    ```python
    from hanami_sdk import user

    address = "http://127.0.0.1:11418"
    user_id = "new_user"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    user.delete_user(token, address, user_id)
    ```

### Add project to user

Assigne a project to a normal user.

The `role` is uses be the policy-file of the Hanami-instance restrict access to specific API-endpoints. Per default there exist `admin` and `member` as roles.

If `is_project_admin` is set to true, the user can access all resources of all users within the project.

=== "Python"

    ```python
    from hanami_sdk import user

    address = "http://127.0.0.1:11418"
    user_id = "new_user"
    project_id = "test_project"
    role = "member"
    is_project_admin = True

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    result = add_roject_to_user(token, 
                                address, 
                                user_id, 
                                project_id, 
                                role, 
                                is_project_admin)
    ```

### Remove project from user

Unassign a project from a user.

=== "Python"

    ```python
    from hanami_sdk import user

    address = "http://127.0.0.1:11418"
    user_id = "new_user"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    result = remove_project_fromUser(token, address, user_id, project_id)
    ```

### List projects of current user

List projects only of the current user, which are enabled by the current token.

=== "Python"

    ```python
    from hanami_sdk import user

    address = "http://127.0.0.1:11418"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    result = list_projects_of_user(token, address)
    ```

### Switch project-scrope of current user

Switch to another project with the current user.

=== "Python"

    ```python
    from hanami_sdk import user

    address = "http://127.0.0.1:11418"
    project_id = "test_project"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    result = switch_project(token, address, project_id)
    ```


## Data-Set

Datasets are a bunch of train- or test-data, which can be uploaded to the server.

### Upload MNIST-Data-Set

These are files of the official mnist-dataset, which can be uploaded and which are primary used for testing currently. Each dataset of this type requires the file-path to the local input- and label-file of the same dataset. 

!!! warning

    Because of a lack of validation at the moment, it is easy to break the backend with unexpected input.

=== "Python"

    ```python
    from hanami_sdk import dataset

    address = "http://127.0.0.1:11418"
    train_dataset_name = "train_test_dataset"
    train_inputs = "/tmp/mnist/train-images.idx3-ubyte"
    train_labels = "/tmp/mnist/train-labels.idx1-ubyte"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    dataset_uuid = dataset.upload_mnist_files(token, address, train_dataset_name, train_inputs, train_labels)

    # example-content of dataset_uuid:
    #
    # 6f2bbcd2-7081-4b08-ae1d-16e6cd6f54c4
    ```

### Upload CSV-Data-Set

!!! warning

    Because of a lack of validation at the moment, it is easy to break the backend with unexpected input.

=== "Python"

    ```python
    from hanami_sdk import dataset

    address = "http://127.0.0.1:11418"
    train_dataset_name = "train_test_dataset"
    train_inputs = "/tmp/csv/test-file.csv"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    dataset_uuid = dataset.upload_csv_files(token, address, train_dataset_name, train_inputs)

    # example-content of dataset_uuid:
    #
    # 6f2bbcd2-7081-4b08-ae1d-16e6cd6f54c4
    ```

### Get Data-Set

Get information about a specific dataset.

=== "Python"

    ```python
    from hanami_sdk import dataset

    address = "http://127.0.0.1:11418"
    dataset_uuid = "6f2bbcd2-7081-4b08-ae1d-16e6cd6f54c4"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    result = dataset.get_dataset(token, address, dataset_uuid)

    # example-content of result:
    #
    # {
    #     "inputs": 784,
    #     "lines": 60000,
    #     "location": "/etc/hanami/datasets/6f2bbcd2-7081-4b08-ae1d-16e6cd6f54c4_mnist_asdf",
    #     "name": "train_test_dataset",
    #     "outputs": 10,
    #     "owner_id": "asdf",
    #     "project_id": "admin",
    #     "type": "mnist",
    #     "uuid": "6f2bbcd2-7081-4b08-ae1d-16e6cd6f54c4",
    #     "visibility": "private"
    # }
    ```

### List Data-Sets

List all visible datasets.

=== "Python"

    ```python
    from hanami_sdk import dataset

    address = "http://127.0.0.1:11418"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    result = dataset.list_datasets(token, address)

    # example-content of result:
    #
    # {
    #     "body": [
    #         [
    #             "6f2bbcd2-7081-4b08-ae1d-16e6cd6f54c4",
    #             "admin",
    #             "asdf",
    #             "private",
    #             "train_test_dataset",
    #             "mnist"
    #         ]
    #     ],
    #     "header": [
    #         "uuid",
    #         "project_id",
    #         "owner_id",
    #         "visibility",
    #         "name",
    #         "type"
    #     ]
    # }
    ```

### Delete Data-Set

Delete a dataset.

=== "Python"

    ```python
    from hanami_sdk import dataset

    address = "http://127.0.0.1:11418"
    dataset_uuid = "6f2bbcd2-7081-4b08-ae1d-16e6cd6f54c4"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    dataset.delete_dataset(token, address, dataset_uuid)
    ```

## Cluster

Cluster containing the neural network.

### Create Cluster

To initialize a new cluster, a cluster-templated is used, which describes the basic structure of the network (see documentation of the [cluster-templates](https://docs.hanami-ai.com/api/cluster_template/))

=== "Python"

    ```python
    from hanami_sdk import cluster

    address = "http://127.0.0.1:11418"
    cluster_name = "test_cluster"
    cluster_template = \
        "version: 1\n" \
        "bricks:\n" \
        "    1,1,1\n" \
        "        input: test_input\n" \
        "        number_of_neurons: 784\n" \
        "    2,1,1\n" \
        "        number_of_neurons: 400\n" \
        "    3,1,1\n" \
        "        output: test_output\n" \
        "        number_of_neurons: 10"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    result = cluster.create_cluster(token, address, cluster_name, cluster_template)

    # example-content of result:
    #
    # {
    #     "name": "test_cluster",
    #     "owner_id": "asdf",
    #     "project_id": "admin",
    #     "uuid": "d94f2b53-f404-4215-9a33-63c4a03e3202",
    #     "visibility": "private"
    # }
    ```


### Get Cluster

Get information of a specific cluster. 

!!! info
    It is basically the same output like coming from the create command and contains only the data stored in the database. Information about the cluster itself, like number of neurons, amount of used memory and so on are still missing in this output currently.

=== "Python"

    ```python
    from hanami_sdk import cluster

    address = "http://127.0.0.1:11418" 
    cluster_uuid = "d94f2b53-f404-4215-9a33-63c4a03e3202"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    result = cluster.get_cluster(token, address, cluster_uuid)

    # example-content of result:
    #
    # {
    #     "name": "test_cluster",
    #     "owner_id": "asdf",
    #     "project_id": "admin",
    #     "uuid": "d94f2b53-f404-4215-9a33-63c4a03e3202",
    #     "visibility": "private"
    # }
    ```


### List Cluster

List all visible cluster.

=== "Python"

    ```python
    from hanami_sdk import cluster

    address = "http://127.0.0.1:11418"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    result = cluster.list_clusters(token, address)
        
    # example-content of result:
    #
    # {
    #     "body": [
    #         [
    #             "d94f2b53-f404-4215-9a33-63c4a03e3202",
    #             "admin",
    #             "asdf",
    #             "private",
    #             "test_cluster"
    #         ]
    #     ],
    #     "header": [
    #         "uuid",
    #         "project_id",
    #         "owner_id",
    #         "visibility",
    #         "name"
    #     ]
    # }
    ```


### Delete Cluster

Delete a cluster from a backend.

=== "Python"

    ```python
    from hanami_sdk import cluster

    address = "http://127.0.0.1:11418"
    cluster_uuid = "d94f2b53-f404-4215-9a33-63c4a03e3202"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    cluster.delete_cluster(token, address, cluster_uuid)
    ```


### Create Checkpoint of Cluster

Save the state of the cluster by creating a checkpoint, which is stored on the server.

=== "Python"

    ```python
    from hanami_sdk import cluster

    address = "http://127.0.0.1:11418"
    checkpoint_name = "test_checkpoint"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    result = cluster.save_cluster(token, address, checkpoint_name, cluster_uuid)

    # example-content of result:
    #
    # {
    #     "name": "test_checkpoint",
    #     "uuid": "d7130869-520f-4743-8f90-4f17f3382321"
    # }
    ```

### Restore Checkpoint of Cluster

Reset a cluster to the state, which is stored in a specific checkpoint.

=== "Python"

    ```python
    from hanami_sdk import cluster

    address = "http://127.0.0.1:11418"
    checkpoint_uuid = "cc6120c7-cc31-4f17-baee-c6c606f00512"
    cluster_uuid = "d94f2b53-f404-4215-9a33-63c4a03e3202"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    result = cluster.restore_cluster(token, address, checkpoint_uuid, cluster_uuid)

    # example-content of result:
    #
    # {
    #     "uuid": "e27e4834-64df-4db2-883e-9bfb44bc6753"
    # }
    ```


### Switch Host

Each CPU and GPU is handled as its logical host. Cluster and be moved between them. To list avaialble hosts there is the [list-hosts endpoint](https://docs.hanami-ai.com/api/sdk_library/#list-hosts).

!!! warning

    Only supported for 1 cpu currently. Support for NUMA-architecture comes in the future. Multiple gpu's are theoretically supported, but this case was not tested currently.

=== "Python"

    ```python
    from hanami_sdk import cluster

    address = "http://127.0.0.1:11418"
    host_uuid = "cc6120c7-cc31-4f17-baee-c6c606f00512"
    cluster_uuid = "d94f2b53-f404-4215-9a33-63c4a03e3202"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    result = cluster.switch_host(token, address, cluster_uuid, host_uuid)

    # example-content of result:
    #
    # {
    #     "name": "test_cluster",
    #     "owner_id": "asdf",
    #     "project_id": "admin",
    #     "uuid": "d94f2b53-f404-4215-9a33-63c4a03e3202",
    #     "visibility": "private"
    # }
    ```

## Task

Tasks are asynchronous actions, which are placed within a queue of the cluster, which should be affected by the task. They are processed one after another. 

### Create Train-Task

Create a new task to train the cluster with the data of a dataset, which was uploaded before.

=== "Python"

    ```python
    from hanami_sdk import task

    address = "http://127.0.0.1:11418"
    task_name = "test_task"
    cluster_uuid = "9f86921d-9a7c-44a2-836c-1683928d9354"
    dataset_uuid = "6f2bbcd2-7081-4b08-ae1d-16e6cd6f54c4"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    result = task.create_task(token, 
                              address, 
                              task_name, 
                              "train", 
                              cluster_uuid, 
                              dataset_uuid)
    task_uuid = json.loads(result)["uuid"]

    # optional you can wait until the task is finished
    finished = False
    while not finished:
        result = task.get_task(token, address, task_uuid, cluster_uuid)
        finished = json.loads(result)["state"] == "finished"
        print("wait for finish task")
        time.sleep(1)
    ```

### Create Request-Task

Create a new task to request information from a trained cluster. As input the data of a dataset are used, which had to be uplaoded first. The output is written into a [request-result](https://docs.hanami-ai.com/api/sdk_library/#request-result). This output has the same UUID and name, like the original task.

=== "Python"

    ```python
    from hanami_sdk import task

    address = "http://127.0.0.1:11418"
    task_name = "test_task"
    cluster_uuid = "9f86921d-9a7c-44a2-836c-1683928d9354"
    dataset_uuid = "6f2bbcd2-7081-4b08-ae1d-16e6cd6f54c4"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    result = task.create_task(token, 
                              address, 
                              task_name, 
                              "request", 
                              cluster_uuid, 
                              dataset_uuid)
    task_uuid = json.loads(result)["uuid"]

    # optional you can wait until the task is finished
    finished = False
    while not finished:
        result = task.get_task(token, address, task_uuid, cluster_uuid)
        finished = json.loads(result)["state"] == "finished"
        print("wait for finish task")
        time.sleep(1)
    ```

### Get Task

=== "Python"

    ```python
    from hanami_sdk import task

    address = "http://127.0.0.1:11418"
    cluster_uuid = "9f86921d-9a7c-44a2-836c-1683928d9354"
    task_uuid = "c7f7e274-5d7d-4696-8591-18441cb1b685"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    result = task.get_task(token, address, task_uuid, cluster_uuid)

    # example-content of result:
    #
    # {
    #     "end_timestamp": "-",
    #     "percentage_finished": 0.6904833316802979,
    #     "queue_timestamp": "2024-01-08 21:19:16",
    #     "start_timestamp": "2024-01-08 21:19:16",
    #     "state": "active"
    # }
    ```

### List Task

List all tasks for a cluster, together with their progress.

=== "Python"

    ```python
    from hanami_sdk import task

    address = "http://127.0.0.1:11418"
    cluster_uuid = "9f86921d-9a7c-44a2-836c-1683928d9354"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    result = task.list_tasks(token, address, cluster_uuid)

    # example-content of result:
    #
    # {
    #     "body": [
    #         [
    #             "ef2ee9e9-d724-49bb-b656-2d5e3484b9f3",
    #             "finished",
    #             "1.000000",
    #             "2024-01-08 21:19:23",
    #             "2024-01-08 21:19:23",
    #             "2024-01-08 21:19:23"
    #         ]
    #     ],
    #     "header": [
    #         "uuid",
    #         "state",
    #         "percentage",
    #         "queued",
    #         "start",
    #         "end"
    #     ]
    # }
    ```


### Delete Task

Delete a task from a cluster. In this task was a request and produced a request-result, this result will not be deleted.

=== "Python"

    ```python
    from hanami_sdk import task

    address = "http://127.0.0.1:11418"
    cluster_uuid = "9f86921d-9a7c-44a2-836c-1683928d9354"
    task_uuid = "c7f7e274-5d7d-4696-8591-18441cb1b685"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    task.delete_task(token, address, task_uuid, cluster_uuid)
    ```


## Direct-IO

It is possible to directly connect via websocket to the cluster on the server to make single requests much faster, because it doesn't use the REST-API. It is an alternative to the tasks. 

!!! info

    To avoid conflicts, the activation of these direct interaction requires an empty task-queue of the related cluster.

!!! warning

    The websocket-connection is not bound to the expire-time of the token.

### direct train

=== "Python"

    ```python
    from hanami_sdk import direct_io

    address = "http://127.0.0.1:11418"
    cluster_uuid = "9f86921d-9a7c-44a2-836c-1683928d9354"
    input_values = [0.0, 2.0, 0.0, 10.0, 0.5]
    exprected_values = [1.0, 0.0]

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    # initial request of a websocket connection to a specific cluster
    # this websocket can be used for multiple request
    ws = cluster.switch_to_direct_mode(token, address, cluster_uuid)

    # names "test_input" and "test_output" are the names of the bricks within the cluster
    # for the mapping of the input-data
    # if the last argument is set to "True", it says that there are no more data to 
    # apply to the cluster and that the train-process can start
    direct_io.send_train_input(ws, "test_input", input_values, False)
    direct_io.send_train_input(ws, "test_output", exprected_values, True)

    cluster.switch_to_task_mode(token, address, cluster_uuid)

    ws.close()
    ```

### direct request

=== "Python"

    ```python
    from hanami_sdk import direct_io

    address = "http://127.0.0.1:11418"
    cluster_uuid = "9f86921d-9a7c-44a2-836c-1683928d9354"
    input_values = [0.0, 2.0, 0.0, 10.0, 0.5]

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    # initial request of a websocket connection to a specific cluster
    # this websocket can be used for multiple request
    ws = cluster.switch_to_direct_mode(token, address, cluster_uuid)

    # names "test_input" is the names of the bricks within the cluster
    # for the mapping of the input-data
    # if the last argument is set to "True", it says that there are no more data to 
    # apply to the cluster and that the train-process can start
    output_values = direct_io.send_request_input(ws, "test_input", input_values, True)

    # output_values is an array like this:
    #    [0.8, 0.0]

    cluster.switch_to_task_mode(token, address, cluster_uuid)

    ws.close()
    ```


## Request-Result

Outputs, which are produced by a request-task, are automatically stored at the end of the task as request-result under the same UUID and name, like the task, which produced the result.

### Get Request-Result

Get the result of a request-task with all of the resulting values.

=== "Python"

    ```python
    from hanami_sdk import request_result

    address = "http://127.0.0.1:11418"
    task_uuid = "c7f7e274-5d7d-4696-8591-18441cb1b685"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    result = request_result.get_request_result(token, address, task_uuid)

    # example-content of result:
    #
    # {
    #     "name": "test_task",
    #     "owner_id": "asdf",
    #     "project_id": "admin",
    #     "uuid": "d40c0c06-bd28-49a4-b872-6a70c4750bb9",
    #     "visibility": "private",
    #     "data": [
    #         1,
    #         2,
    #         8,
    #         4,
    #         5,
    #         6,
    #         7,
    #         8,
    #         9,
    #         0,
    #         1,
    #         2,
    #         3,
    #         4,
    #         5,
    #         6
    #     ]
    # }
    ```

### List Request-Result

List all visible request-results.

=== "Python"

    ```python
    from hanami_sdk import request_result

    address = "http://127.0.0.1:11418"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    result = request_result.list_request_results(token, address)

    # example-content of result:
    #
    # {
    #     "body": [
    #         [
    #             "d40c0c06-bd28-49a4-b872-6a70c4750bb9",
    #             "admin",
    #             "asdf",
    #             "private",
    #             "test_task"
    #         ]
    #     ],
    #     "header": [
    #         "uuid",
    #         "project_id",
    #         "owner_id",
    #         "visibility",
    #         "name"
    #     ]
    # }
    ```

### Delete Request-Result

Delete a request-result.

=== "Python"

    ```python
    from hanami_sdk import request_result

    address = "http://127.0.0.1:11418"
    task_uuid = "c7f7e274-5d7d-4696-8591-18441cb1b685"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    request_result.delete_request_result(token, address, task_uuid)
    ```

### Check Dataset

Checks a request-result against a dataset to compare who much of the output of the network was correct. The output gives the percentage of the correct output-values.

=== "Python"

    ```python
    from hanami_sdk import request_result

    address = "http://127.0.0.1:11418"
    task_uuid = "c7f7e274-5d7d-4696-8591-18441cb1b685"
    request_dataset_uuid = "d40c0c06-bd28-49a4-b872-6a70c4750bb9"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    result = request_result.check_against_dataset(token, 
                                                  address, 
                                                  task_uuid, 
                                                  request_dataset_uuid)

    # example-content of result:
    #
    # {
    #     "accuracy": 93.40999603271484
    # }
    ```


## Checkpoint

Checkpoints are a copy of the current state of a cluster. It can be used as backup to restore an older state a cluster.

!!! info

    It is possible to apply a checkpoint to any cluster, but at the moment it is not possible to directly create a new cluster out of a checkpoint. 

### List Checkpoints

List all visible checkpoints.

=== "Python"

    ```python
    from hanami_sdk import checkpoint

    address = "http://127.0.0.1:11418"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    result = checkpoint.list_checkpoints(token, address)

    # example-content of result:
    #
    # {
    #     "body": [
    #         [
    #             "cc6120c7-cc31-4f17-baee-c6c606f00512",
    #             "admin",
    #             "asdf",
    #             "private",
    #             "test_checkpoint"
    #         ]
    #     ],
    #     "header": [
    #         "uuid",
    #         "project_id",
    #         "owner_id",
    #         "visibility",
    #         "name"
    #     ]
    # }
    ```

### Delete Checkpoint

Delete a checkpoint from the backend.

=== "Python"

    ```python
    from hanami_sdk import checkpoint

    address = "http://127.0.0.1:11418"
    checkpoint_uuid = "cc6120c7-cc31-4f17-baee-c6c606f00512"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    checkpoint.delete_checkpoint(token, address, checkpoint_uuid)

    ```


## Hosts

### List Hosts

Each CPU and GPU is handled as its own logical host to have more control over the exact location of the data. These logical hosts can be listed with this endpoint.

=== "Python"

    ```python
    from hanami_sdk import hosts

    address = "http://127.0.0.1:11418"

    # request a token for a user, who has admin-permissions
    # see: https://docs.hanami-ai.com/api/sdk_library/#request-token

    result = hosts.list_hosts(token, address)

    # example-content of result:
    #
    # {
    #     "body": [
    #         [
    #             "cc6120c7-cc31-4f17-baee-c6c606f00512",
    #             "cpu",
    #         ]
    #     ],
    #     "header": [
    #         "uuid",
    #         "type"
    #     ]
    # }
    ```
