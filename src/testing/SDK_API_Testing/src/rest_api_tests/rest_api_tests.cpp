/**
 * @file        rest_api_tests.cpp
 *
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

#include "rest_api_tests.h"

#include <libKitsunemimiConfig/config_handler.h>
#include <libKitsunemimiJson/json_item.h>
#include <libHanamiAiSdk/init.h>

#include <common/test_thread.h>
#include <libHanamiAiSdk/cluster.h>
#include <libHanamiAiSdk/user.h>
#include <libHanamiAiSdk/project.h>

#include <rest_api_tests/misaki/project/project_create_test.h>
#include <rest_api_tests/misaki/project/project_delete_test.h>
#include <rest_api_tests/misaki/project/project_get_test.h>
#include <rest_api_tests/misaki/project/project_list_test.h>

#include <rest_api_tests/misaki/user/user_create_test.h>
#include <rest_api_tests/misaki/user/user_delete_test.h>
#include <rest_api_tests/misaki/user/user_get_test.h>
#include <rest_api_tests/misaki/user/user_list_test.h>

#include <rest_api_tests/shiori/datasets/dataset_create_mnist_test.h>
#include <rest_api_tests/shiori/datasets/dataset_create_csv_test.h>
#include <rest_api_tests/shiori/datasets/dataset_delete_test.h>
#include <rest_api_tests/shiori/datasets/dataset_get_test.h>
#include <rest_api_tests/shiori/datasets/dataset_list_test.h>
#include <rest_api_tests/shiori/datasets/dataset_check_test.h>

#include <rest_api_tests/shiori/snapshots/snapshot_delete_test.h>
#include <rest_api_tests/shiori/snapshots/snapshot_get_test.h>
#include <rest_api_tests/shiori/snapshots/snapshot_list_test.h>

#include <rest_api_tests/shiori/request_results/request_result_get_test.h>
#include <rest_api_tests/shiori/request_results/request_result_list_test.h>
#include <rest_api_tests/shiori/request_results/request_result_delete_test.h>

#include <rest_api_tests/kyouko/cluster/cluster_create_test.h>
#include <rest_api_tests/kyouko/cluster/cluster_delete_test.h>
#include <rest_api_tests/kyouko/cluster/cluster_get_test.h>
#include <rest_api_tests/kyouko/cluster/cluster_list_test.h>
#include <rest_api_tests/kyouko/cluster/cluster_save_test.h>
#include <rest_api_tests/kyouko/cluster/cluster_load_test.h>
#include <rest_api_tests/kyouko/cluster/cluster_switch_to_direct_test.h>
#include <rest_api_tests/kyouko/cluster/cluster_switch_to_task_test.h>

#include <rest_api_tests/kyouko/io/direct_io_test.h>

#include <rest_api_tests/kyouko/task/image_learn_task_test.h>
#include <rest_api_tests/kyouko/task/image_request_task_test.h>
#include <rest_api_tests/kyouko/task/table_learn_task_test.h>
#include <rest_api_tests/kyouko/task/table_request_task_test.h>

/**
 * @brief initialize client by requesting a token, which is used for all tests
 */
bool
initClient()
{
    Kitsunemimi::ErrorContainer error;

    bool success = false;
    const std::string host = GET_STRING_CONFIG("connection", "host", success);
    const std::string port = std::to_string(GET_INT_CONFIG("connection", "port", success));
    const std::string user = GET_STRING_CONFIG("connection", "test_user", success);
    const std::string pw = GET_STRING_CONFIG("connection", "test_pw", success);

    if(HanamiAI::initClient(host, port, user, pw, error) == false)
    {
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief delete all templates of the test-user to avoid name-conflics
 */
void
deleteAllClusters()
{
    std::string result = "";
    Kitsunemimi::ErrorContainer error;
    HanamiAI::listCluster(result, error);

    Kitsunemimi::JsonItem parsedList;
    parsedList.parse(result, error);

    Kitsunemimi::JsonItem body = parsedList.get("body");
    for(uint64_t i = 0; i < body.size(); i++)
    {
        const std::string uuid = body.get(i).get(0).getString();
        HanamiAI::deleteCluster(result, uuid, error);
    }
}

/**
 * @brief delete all templates of the test-user to avoid name-conflics
 */
void
deleteAllProjects()
{
    std::string result = "";
    Kitsunemimi::ErrorContainer error;
    HanamiAI::listProject(result, error);

    Kitsunemimi::JsonItem parsedList;
    parsedList.parse(result, error);

    Kitsunemimi::JsonItem body = parsedList.get("body");
    for(uint64_t i = 0; i < body.size(); i++)
    {
        const std::string uuid = body.get(i).get(0).getString();
        HanamiAI::deleteProject(result, uuid, error);
    }
}

/**
 * @brief delete all templates of the test-user to avoid name-conflics
 */
void
deleteAllUsers()
{
    std::string result = "";
    Kitsunemimi::ErrorContainer error;
    HanamiAI::listUser(result, error);

    Kitsunemimi::JsonItem parsedList;
    parsedList.parse(result, error);

    Kitsunemimi::JsonItem body = parsedList.get("body");
    for(uint64_t i = 0; i < body.size(); i++)
    {
        const std::string uuid = body.get(i).get(0).getString();
        HanamiAI::deleteUser(result, uuid, error);
    }
}

/**
 * @brief run tests with the usage of MNIST-images
 *
 * @param inputData json-item with names and other predefined values for the tests
 */
void
runImageTest(Kitsunemimi::JsonItem &inputData)
{
    deleteAllClusters();
    deleteAllProjects();
    deleteAllUsers();

    TestThread testThread("test_thread", inputData);

    Kitsunemimi::JsonItem overrideData;

    // test project in misaki
    testThread.addTest(new ProjectCreateTest(true));
    testThread.addTest(new ProjectCreateTest(false));
    testThread.addTest(new ProjectListTest(true));
    testThread.addTest(new ProjectGetTest(true));
    testThread.addTest(new ProjectGetTest(false, "fail_project"));

    // test user in misaki
    testThread.addTest(new UserCreateTest(true));
    testThread.addTest(new UserCreateTest(false));
    testThread.addTest(new UserListTest(true));
    testThread.addTest(new UserGetTest(true));
    testThread.addTest(new UserGetTest(false, "fail_user"));

    // test data-sets of shiori
    testThread.addTest(new DataSetCreateMnistTest(true, "request"));
    testThread.addTest(new DataSetCreateMnistTest(true, "learn"));
    testThread.addTest(new DataSetListTest(true));
    testThread.addTest(new DataSetGetTest(true, "learn"));
    testThread.addTest(new DataSetGetTest(false, "learn", "fail_user"));

    // test cluster of kyouko
    testThread.addTest(new ClusterCreateTest(true));
    testThread.addTest(new ClusterGetTest(true));
    testThread.addTest(new ClusterListTest(true));

    // test learning-tasks of kyouko
    for(int i = 0; i < 1; i++) {
        testThread.addTest(new ImageLearnTaskTest(true));
    }

    // test cluster load and restore of kyouko and shiori
    //testThread.addTest(new ClusterSaveTest(true));
    //testThread.addTest(new ClusterDeleteTest(true));
    //testThread.addTest(new ClusterCreateTest(true));
    //testThread.addTest(new ClusterLoadTest(true));

    // test request-tasks of kyouko
    testThread.addTest(new ImageRequestTaskTest(true));
    testThread.addTest(new DataSetCheckTest(true));

    // test request-result endpoints in shiori
    testThread.addTest(new RequestResultGetTest(true));
    testThread.addTest(new RequestResultListTest(true));

    // test snapshots of shiori
    //testThread.addTest(new SnapshotGetTest(true));
    //testThread.addTest(new SnapshotListTest(true));

    // test direct-io of kyouko
    testThread.addTest(new ClusterSwitchToDirectTest(true));
    testThread.addTest(new DirectIoTest(true));
    testThread.addTest(new ClusterSwitchToTaskTest(true));

    // test delete of all
    testThread.addTest(new UserDeleteTest(true));
    testThread.addTest(new UserDeleteTest(false));
    testThread.addTest(new ProjectDeleteTest(true));
    testThread.addTest(new ProjectDeleteTest(false));
    //testThread.addTest(new SnapshotDeleteTest(true));
    testThread.addTest(new ClusterDeleteTest(true));
    testThread.addTest(new ClusterDeleteTest(false));
    testThread.addTest(new RequestResultDeleteTest(true));
    testThread.addTest(new RequestResultDeleteTest(false));
    testThread.addTest(new DataSetDeleteTest(true, "request"));
    testThread.addTest(new DataSetDeleteTest(true, "learn"));
    testThread.addTest(new DataSetDeleteTest(false, "learn"));

    // check that the running user can not delete himself
    bool success = false;
    const std::string user = GET_STRING_CONFIG("connection", "test_user", success);
    testThread.addTest(new UserDeleteTest(false, user));

    testThread.startThread();

    while(testThread.isFinished == false) {
        usleep(100000);
    }
}

/**
 * @brief runRestApiTests
 */
bool
runRestApiTests()
{
    bool success = false;

    if(initClient() == false) {
        return false;
    }

    const std::string clusterDefinition("version: 1\n"
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

    Kitsunemimi::JsonItem inputData;

    // add data for the test-user to create
    inputData.insert("user_id", "tsugumi");
    inputData.insert("user_name", "Tsugumi");
    inputData.insert("password", "new password");
    inputData.insert("admin", true);
    inputData.insert("role", "tester");
    inputData.insert("project_id", "test_project");
    inputData.insert("project_name", "Test Project");

    // add data from config
    inputData.insert("learn_inputs", GET_STRING_CONFIG("test_data", "learn_inputs", success)),
    inputData.insert("learn_labels", GET_STRING_CONFIG("test_data", "learn_labels", success)),
    inputData.insert("request_inputs", GET_STRING_CONFIG("test_data", "request_inputs", success)),
    inputData.insert("request_labels", GET_STRING_CONFIG("test_data", "request_labels", success)),
    inputData.insert("base_inputs", GET_STRING_CONFIG("test_data", "base_inputs", success)),

    // add predefined names for the coming test-resources
    inputData.insert("cluster_name", "test_cluster");
    inputData.insert("cluster_snapshot_name", "test_snapshot");
    inputData.insert("generic_task_name", "test_task");
    inputData.insert("template_name", "dynamic");
    inputData.insert("cluster_definition", clusterDefinition);
    inputData.insert("request_dataset_name", "request_test_dataset");
    inputData.insert("learn_dataset_name", "learn_test_dataset");
    inputData.insert("base_dataset_name", "base_test_dataset");

    runImageTest(inputData);

    return true;
}

