# SDK-library

Repo: [hanami_sdk](https://github.com/kitsudaiki/hanami_sdk)

The SDK-library privides functions to interact with the API of the backend. At the moment 2 versions are provided:

- `C++`
- (`Javascript`)

!!! info

    Because of a lack of time the `Javascript`-version is a bit incomplete in its implementation and not in this documentation here. This will be updated later.

!!! info

    There is also some code for `go` in the repository. This is old outdated code, which was added for the PoC of an in `go` written CLI-client. This will be condinued, but at the moment this code doesn't work and is incomplete.


## Login


=== "C++"

    ``` c++
    #include <hanami_sdk/init.h>

    Hanami::ErrorContainer error;
    const std::string host = "127.0.0.1";
    const std::string port = "1337";
    const std::string user = "test_user";
    const std::string password = "12345";

    if(HanamiAI::initClient(host, port, user, pw, error))
    {
        // action if successful
    } 
    else 
    {
        std::cout<<error.toString()<<std::endl;
    }
    ```

## User

### Create User

=== "C++"

    ``` c++
    #include <hanami_sdk/user.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string userId = "test_user";
    const std::string userName = "Test User";
    const std::string password = "12345";
    const bool isAdmin = false;

    if(HanamiAI::createUser(result, userId, userName, password, isAdmin, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }
    ```

### Get User

=== "C++"

    ``` c++
    #include <hanami_sdk/user.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string userId = "test_user";

    if(HanamiAI::getUser(result, userId, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }
    ```

### List User

=== "C++"

    ``` c++
    #include <hanami_sdk/user.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string userId = "test_user";

    if(HanamiAI::getUser(result, userId, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }
    ```

### Delete User

=== "C++"

    ``` c++
    #include <hanami_sdk/user.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string userId = "test_user";

    if(HanamiAI::getUser(result, userId, error) == false) {
        std::cout<<error.toString()<<std::endl;
    }
    ```

### Add project to user

=== "C++"

    ``` c++
    #include <hanami_sdk/user.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string userId = "test_user";
    const std::string projectId = "test_project";
    const std::string role = "member";
    const bool isProjectAdmin = false;

    if(HanamiAI::addProjectToUser(result, userId, projectId, role, isProjectAdmin, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }
    ```

### Remove project from user

=== "C++"

    ``` c++
    #include <hanami_sdk/user.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string userId = "test_user";
    const std::string projectId = "test_project";

    if(HanamiAI::removeProjectFromUser(result, userId, projectId, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }
    ```

### List projects of current user

=== "C++"

    ``` c++
    #include <hanami_sdk/user.h>

    std::string result;
    Hanami::ErrorContainer error;

    if(HanamiAI::listProjectsOfUser(result, userId, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }
    ```

### Switch project-scrope of current user

=== "C++"

    ``` c++
    #include <hanami_sdk/user.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string projectId = "test_project";

    if(HanamiAI::switchProject(result, projectId, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }
    ```



## Project

### Create Project

=== "C++"

    ``` c++
    #include <hanami_sdk/project.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string projectId = "test_project";
    const std::string projectName = "Test Project";

    if(HanamiAI::createProject(result, projectId, projectName, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }                     
    ```

### Get Project

=== "C++"

    ``` c++
    #include <hanami_sdk/project.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string projectId = "test_project";

    if(HanamiAI::getProject(result, projectId, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    } 
    ```

### List Project

=== "C++"

    ``` c++
    #include <hanami_sdk/project.h>

    std::string result;
    Hanami::ErrorContainer error;

    if(HanamiAI::listProject(result, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    } 
    ```

### Delete Project

=== "C++"

    ``` c++
    #include <hanami_sdk/project.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string projectId = "test_project";

    if(HanamiAI::deleteProject(result, projectId, error)== false) {
        std::cout<<error.toString()<<std::endl;
    }
    ```


## Data-Set

### Upload CSV-Data-Set

=== "C++"

    ``` c++
    #include <hanami_sdk/data_set.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string datasetName = "Test Dataset";
    const std::string csvFilePath = "/tmp/test.csv";

    if(HanamiAI::uploadCsvData(result, datasetName, csvFilePath, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }
    ```

### Upload MNIST-Data-Set

=== "C++"

    ``` c++
    #include <hanami_sdk/data_set.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string datasetName = "Test Dataset";
    const std::string inputFilePath = "/tmp/train-images.idx3-ubyte";
    const std::string labelFilePath = "/tmp/train-labels.idx1-ubyte";

    if(HanamiAI::uploadCsvData(result, datasetName, inputFilePath, labelFilePath, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }
    ```

### Get Data-Set

=== "C++"

    ``` c++
    #include <hanami_sdk/data_set.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string datasetUuid = "30cfeb32-48d7-11ed-b878-0242ac120002";

    if(HanamiAI::getDataset(result, datasetUuid, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }
    ```

### List Data-Sets

=== "C++"

    ``` c++
    #include <hanami_sdk/data_set.h>

    std::string result;
    Hanami::ErrorContainer error;

    if(HanamiAI::listDatasets(result, requestUuid, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }
    ```

### Delete Data-Set

=== "C++"

    ``` c++
    #include <hanami_sdk/data_set.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string datasetUuid = "30cfeb32-48d7-11ed-b878-0242ac120002";

    if(HanamiAI::deleteDataset(result, datasetUuid, error) == false) {
        std::cout<<error.toString()<<std::endl;
    }
    ```

### Check Data-Set

Compares a `Request-Result` against a `Data-Set` to check, how much of the Data-Set was correctly identified by the request-task, to check the quality of the learning-processing.

!!! warning

    This endpoint was until now only used in order to check the request of the MNIST-test. It will not work with CSV-Data-Sets.

=== "C++"

    ``` c++
    #include <hanami_sdk/data_set.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string requestUuid = "d922013a-48d2-11ed-b878-0242ac120002";
    const std::string datasetUuid = "30cfeb32-48d7-11ed-b878-0242ac120002";

    if(HanamiAI::checkDataset(result, datasetUuid, requestUuid, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }
    ```

## Cluster

### Create Cluster

=== "C++"

    ``` c++
    #include <hanami_sdk/cluster.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string clusterName = "test_cluster";
    const std::string clusterTemplate = ...;

    if(HanamiAI::createCluster(result, clusterName, clusterTemplate, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }
    ```

    ??? example "Example for clusterTemplate"

        ```
        const std::string clusterTemplate = "version: 1\n"
                                            "settings:\n"
                                            "    max_synapse_sections: 1000\n"
                                            "    sign_neg: 0.5\n"
                                            "        \n"
                                            "bricks:\n"
                                            "    1,1,1\n"
                                            "        input: test_input\n"
                                            "        number_of_neurons: 784\n"
                                            "    2,1,1\n"
                                            "        number_of_neurons: 400\n"
                                            "    3,1,1\n"
                                            "        output: test_output\n"
                                            "        number_of_neurons: 10");
        ```


### Get Cluster


=== "C++"

    ``` c++
    #include <hanami_sdk/cluster.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string clusterUuid = "d922013a-48d2-11ed-b878-0242ac120002";

    if(HanamiAI::getCluster(result, clusterUuid, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }
    ```


### List Cluster


=== "C++"

    ``` c++
    #include <hanami_sdk/cluster.h>

    std::string result;
    Hanami::ErrorContainer error;

    if(HanamiAI::listCluster(result, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }
    ```


### Delete Cluster


=== "C++"

    ``` c++
    #include <hanami_sdk/cluster.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string clusterUuid = "d922013a-48d2-11ed-b878-0242ac120002";

    if(HanamiAI::deleteCluster(result, clusterUuid, error) == false) {
        std::cout<<error.toString()<<std::endl;
    }
    ```


### Create Checkpoint of Cluster

=== "C++"

    ``` c++
    #include <hanami_sdk/cluster.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string clusterUuid = "d922013a-48d2-11ed-b878-0242ac120002";
    const std::string checkpointName = "test-checkpoint";

    if(HanamiAI::saveCluster(result, clusterUuid, checkpointName, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }
    ```

### Restore Checkpoint of Cluster

=== "C++"

    ``` c++
    #include <hanami_sdk/cluster.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string clusterUuid = "d922013a-48d2-11ed-b878-0242ac120002";
    const std::string checkpointUuid = "30cfeb32-48d7-11ed-b878-0242ac120002";

    if(HanamiAI::restoreCluster(result, clusterUuid, checkpointUuid, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }
    ```

### Switch Cluster to Task-Mode

=== "C++"

    ``` c++
    #include <hanami_sdk/cluster.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string clusterUuid = "d922013a-48d2-11ed-b878-0242ac120002";

    if(HanamiAI::switchToTaskMode(result, clusterUuid, error)){
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }
    ```

### Switch Cluster to Direct-Mode

see [Initializing Websocket](/How_To/5_sdk_library/#init-websocket)


## Request-Result

### Get Request-Result

=== "C++"

    ``` c++
    #include <hanami_sdk/request_result.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string resultUuid = "d922013a-48d2-11ed-b878-0242ac120002";

    if(HanamiAI::getRequestResult(result, resultUuid, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }
    ```

### List Request-Result

=== "C++"

    ``` c++
    #include <hanami_sdk/request_result.h>

    std::string result;
    Hanami::ErrorContainer error;

    if(HanamiAI::listRequestResult(result, resultUuid, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }
    ```

### Delete Request-Result

=== "C++"

    ``` c++
    #include <hanami_sdk/request_result.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string resultUuid = "d922013a-48d2-11ed-b878-0242ac120002";

    if(HanamiAI::deleteRequestResult(result, resultUuid, error) == false) {
        std::cout<<error.toString()<<std::endl;
    }
    ```


## Task

### Create Table-Train-Task

=== "C++"

    ``` c++
    #include <hanami_sdk/task.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string clusterUuid = "d922013a-48d2-11ed-b878-0242ac120002";
    const std::string dataSetUuid = "30cfeb32-48d7-11ed-b878-0242ac120002";

    if(HanamiAI::createTableTrainTask(result, name, clusterUuid, dataSetUuid, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }
    ```

### Create Table-Request-Task

=== "C++"

    ``` c++
    #include <hanami_sdk/task.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string clusterUuid = "d922013a-48d2-11ed-b878-0242ac120002";
    const std::string dataSetUuid = "30cfeb32-48d7-11ed-b878-0242ac120002";

    if(HanamiAI::createTableRequestTask(result, name, clusterUuid, dataSetUuid, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }
    ```

### Create Image-Train-Task

=== "C++"

    ``` c++
    #include <hanami_sdk/task.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string clusterUuid = "d922013a-48d2-11ed-b878-0242ac120002";
    const std::string dataSetUuid = "30cfeb32-48d7-11ed-b878-0242ac120002";

    if(HanamiAI::createImageTrainTask(result, name, clusterUuid, dataSetUuid, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }
    ```

### Create Image-Request-Task

=== "C++"

    ``` c++
    #include <hanami_sdk/task.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string clusterUuid = "d922013a-48d2-11ed-b878-0242ac120002";
    const std::string dataSetUuid = "30cfeb32-48d7-11ed-b878-0242ac120002";

    if(HanamiAI::createImageRequestTask(result, name, clusterUuid, dataSetUuid, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }
    ```

### Get Task

=== "C++"

    ``` c++
    #include <hanami_sdk/task.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string taskUuid = "d922013a-48d2-11ed-b878-0242ac120002";
    const std::string clusterUuid = "30cfeb32-48d7-11ed-b878-0242ac120002";

    if(HanamiAI::getTask(result, taskUuid, clusterUuid, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }
    ```

### List Task

=== "C++"

    ``` c++
    #include <hanami_sdk/task.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string clusterUuid = "30cfeb32-48d7-11ed-b878-0242ac120002";

    if(HanamiAI::listTask(result, clusterUuid, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }
    ```

### Delete Task

=== "C++"

    ``` c++
    #include <hanami_sdk/task.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string taskUuid = "d922013a-48d2-11ed-b878-0242ac120002";
    const std::string clusterUuid = "30cfeb32-48d7-11ed-b878-0242ac120002";

    if(HanamiAI::deleteTask(result, taskUuid, clusterUuid, error) == false) {
        std::cout<<error.toString()<<std::endl;
    }
    ```


## Cluster-Checkpoint

### Get Cluster-Checkpoint

=== "C++"

    ``` c++
    #include <hanami_sdk/checkpoint.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string checkpointUuid = "30cfeb32-48d7-11ed-b878-0242ac120002";

    if(HanamiAI::getCheckpoint(result, checkpointUuid, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }
    ```

### List Cluster-Checkpoints

=== "C++"

    ``` c++
    #include <hanami_sdk/checkpoint.h>

    std::string result;
    Hanami::ErrorContainer error;

    if(HanamiAI::listCheckpoint(result, error)) {
        std::cout<<result<<std::endl;
    } else {
        std::cout<<error.toString()<<std::endl;
    }
    ```

### Delete Cluster-Checkpoint

=== "C++"

    ``` c++
    #include <hanami_sdk/checkpoint.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string checkpointUuid = "30cfeb32-48d7-11ed-b878-0242ac120002";

    if(HanamiAI::deleteCheckpoint(result, checkpointUuid, error) == false) {
        std::cout<<error.toString()<<std::endl;
    }
    ```


## Direct-IO

### Init Websocket

=== "C++"

    ``` c++
    #include <hanami_sdk/cluster.h>
    #include <hanami_sdk/common/websocket_client.h>

    std::string result;
    Hanami::ErrorContainer error;
    const std::string clusterUuid = "30cfeb32-48d7-11ed-b878-0242ac120002";

    HanamiAI::WebsocketClient* client = nullptr;
    client = HanamiAI::switchToDirectMode(result, clusterUuid, error);
    if(client != nullptr) {
        // success
    }
    ```

### direct train

=== "C++"

    ``` c++
    #include <hanami_sdk/io.h>

    std::vector<float> input;
    // init number of input-values, which match the number of inputs of the cluster

    std::vector<float> expectedOutput;
    // init number of expected-values, which match the number of outputs of the cluster

    HanamiAI::WebsocketClient* client = // see "Init Websocket" above for client-init

    if(HanamiAI::train(client, inpututValues, expectedOutput, error))
    {
        // success
    }
    ```

### direct request

=== "C++"

    ``` c++
    #include <hanami_sdk/io.h>

    std::vector<float> input;
    // init number of input-values, which match the number of inputs of the cluster

    HanamiAI::WebsocketClient* client = // see "Init Websocket" above for client-init

    uint64_t numberOfValues = 0;
    float* values = HanamiAI::request(client, inpututValues, numberOfValues, error);
    if(values != nullptr) 
    {
        for(uint32_t i = 0; i < numberOfValues; i++) {
            std::cout<<i<<": "<<values[i]<<std::endl;
        }

        delete[] values;
    }
    ```

