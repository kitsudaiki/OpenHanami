/**
 * @file        blossom_initializing.h
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

#ifndef HANAMI_BLOSSOM_INITIALIZING_H
#define HANAMI_BLOSSOM_INITIALIZING_H

#include <common.h>

#include <libKitsunemimiConfig/config_handler.h>
#include <libKitsunemimiCommon/methods/file_methods.h>
#include <libKitsunemimiCommon/files/text_file.h>
#include <libKitsunemimiCommon/logger.h>

#include <api/http/v1/cluster/create_cluster.h>
#include <api/http/v1/cluster/show_cluster.h>
#include <api/http/v1/cluster/list_cluster.h>
#include <api/http/v1/cluster/delete_cluster.h>
#include <api/http/v1/cluster/save_cluster.h>
#include <api/http/v1/cluster/load_cluster.h>
#include <api/http/v1/cluster/set_cluster_mode.h>

#include <api/http/v1/task/create_task.h>
#include <api/http/v1/task/show_task.h>
#include <api/http/v1/task/list_task.h>
#include <api/http/v1/task/delete_task.h>

#include <api/http/v1/system_info/get_system_info.h>

#include <api/http/v1/threading/get_thread_mapping.h>

#include <api/http/v1/measurements/power_consumption.h>
#include <api/http/v1/measurements/temperature_production.h>
#include <api/http/v1/measurements/speed.h>

#include <api/http/v1/data_files/list_data_set.h>
#include <api/http/v1/data_files/get_data_set.h>
#include <api/http/v1/data_files/delete_data_set.h>
#include <api/http/v1/data_files/check_data_set.h>
#include <api/http/v1/data_files/get_progress_data_set.h>
#include <api/http/v1/data_files/mnist/create_mnist_data_set.h>
#include <api/http/v1/data_files/mnist/finalize_mnist_data_set.h>
#include <api/http/v1/data_files/csv/create_csv_data_set.h>
#include <api/http/v1/data_files/csv/finalize_csv_data_set.h>

#include <api/http/v1/checkpoint/create_checkpoint.h>
#include <api/http/v1/checkpoint/delete_checkpoint.h>
#include <api/http/v1/checkpoint/finish_checkpoint.h>
#include <api/http/v1/checkpoint/get_checkpoint.h>
#include <api/http/v1/checkpoint/list_checkpoint.h>

#include <api/http/v1/request_results/delete_request_result.h>
#include <api/http/v1/request_results/get_request_result.h>
#include <api/http/v1/request_results/list_request_result.h>

#include <api/http/v1/logs/get_audit_log.h>
#include <api/http/v1/logs/get_error_log.h>

#include <api/http/v1/user/create_user.h>
#include <api/http/v1/user/get_user.h>
#include <api/http/v1/user/list_users.h>
#include <api/http/v1/user/delete_user.h>
#include <api/http/v1/user/add_project_to_user.h>
#include <api/http/v1/user/remove_project_from_user.h>

#include <api/http/v1/project/create_project.h>
#include <api/http/v1/project/get_project.h>
#include <api/http/v1/project/list_projects.h>
#include <api/http/v1/project/delete_project.h>

#include  <api/http/v1/documentation/generate_rest_api_docu.h>

#include <api/http/v1/auth/create_token.h>
#include <api/http/v1/auth/validate_access.h>
#include <api/http/v1/auth/list_user_projects.h>
#include <api/http/v1/auth/renew_token.h>

#include <hanami_root.h>

/**
 * @brief initClusterBlossoms
 */
void
initClusterBlossoms()
{
    const std::string group = "Cluster";

    assert(HanamiRoot::root->addBlossom(group, "create", new CreateCluster()));
    HanamiRoot::root->addEndpoint("v1/cluster",
                                  Kitsunemimi::Hanami::POST_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "create");

    assert(HanamiRoot::root->addBlossom(group, "show", new ShowCluster()));
    HanamiRoot::root->addEndpoint("v1/cluster",
                                  Kitsunemimi::Hanami::GET_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "show");

    assert(HanamiRoot::root->addBlossom(group, "list", new ListCluster()));
    HanamiRoot::root->addEndpoint("v1/cluster/all",
                                  Kitsunemimi::Hanami::GET_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "list");

    assert(HanamiRoot::root->addBlossom(group, "delete", new DeleteCluster()));
    HanamiRoot::root->addEndpoint("v1/cluster",
                                  Kitsunemimi::Hanami::DELETE_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "delete");

    assert(HanamiRoot::root->addBlossom(group, "save", new SaveCluster()));
    HanamiRoot::root->addEndpoint("v1/cluster/save",
                                  Kitsunemimi::Hanami::POST_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "save");

    assert(HanamiRoot::root->addBlossom(group, "load", new LoadCluster()));
    HanamiRoot::root->addEndpoint("v1/cluster/load",
                                  Kitsunemimi::Hanami::POST_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "load");

    assert(HanamiRoot::root->addBlossom(group, "set_mode", new SetClusterMode()));
    HanamiRoot::root->addEndpoint("v1/cluster/set_mode",
                                  Kitsunemimi::Hanami::PUT_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "set_mode");
}

/**
 * @brief initTaskBlossoms
 */
void
initTaskBlossoms()
{
    const std::string group = "Task";

    assert(HanamiRoot::root->addBlossom(group, "create", new CreateTask()));
    HanamiRoot::root->addEndpoint("v1/task",
                                  Kitsunemimi::Hanami::POST_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "create");

    assert(HanamiRoot::root->addBlossom(group, "show", new ShowTask()));
    HanamiRoot::root->addEndpoint("v1/task",
                                  Kitsunemimi::Hanami::GET_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "show");

    assert(HanamiRoot::root->addBlossom(group, "list", new ListTask()));
    HanamiRoot::root->addEndpoint("v1/task/all",
                                  Kitsunemimi::Hanami::GET_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "list");

    assert(HanamiRoot::root->addBlossom(group, "delete", new DeleteTask()));
    HanamiRoot::root->addEndpoint("v1/task",
                                  Kitsunemimi::Hanami::DELETE_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "delete");
}

/**
 * @brief init data_set blossoms
 */
void
dataSetBlossoms()
{
    const std::string group = "Data Set";

    assert(HanamiRoot::root->addBlossom(group, "create_mnist", new CreateMnistDataSet()));
    HanamiRoot::root->addEndpoint("v1/mnist/data_set",
                                  Kitsunemimi::Hanami::POST_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "create_mnist");

    assert(HanamiRoot::root->addBlossom(group, "finalize_mnist", new FinalizeMnistDataSet()));
    HanamiRoot::root->addEndpoint("v1/mnist/data_set",
                                  Kitsunemimi::Hanami::PUT_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "finalize_mnist");

    assert(HanamiRoot::root->addBlossom(group, "create_csv", new CreateCsvDataSet()));
    HanamiRoot::root->addEndpoint("v1/csv/data_set",
                                  Kitsunemimi::Hanami::POST_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "create_csv");

    assert(HanamiRoot::root->addBlossom(group, "finalize_csv", new FinalizeCsvDataSet()));
    HanamiRoot::root->addEndpoint("v1/csv/data_set",
                                  Kitsunemimi::Hanami::PUT_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "finalize_csv");

    assert(HanamiRoot::root->addBlossom(group, "check", new CheckDataSet()));
    HanamiRoot::root->addEndpoint("v1/data_set/check",
                                  Kitsunemimi::Hanami::POST_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "check");

    assert(HanamiRoot::root->addBlossom(group, "progress", new GetProgressDataSet()));
    HanamiRoot::root->addEndpoint("v1/data_set/progress",
                                  Kitsunemimi::Hanami::GET_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "progress");

    assert(HanamiRoot::root->addBlossom(group, "get", new GetDataSet()));
    HanamiRoot::root->addEndpoint("v1/data_set",
                                  Kitsunemimi::Hanami::GET_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "get");

    assert(HanamiRoot::root->addBlossom(group, "delete", new DeleteDataSet()));
    HanamiRoot::root->addEndpoint("v1/data_set",
                                  Kitsunemimi::Hanami::DELETE_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "delete");

    assert(HanamiRoot::root->addBlossom(group, "list", new ListDataSet()));
    HanamiRoot::root->addEndpoint("v1/data_set/all",
                                  Kitsunemimi::Hanami::GET_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "list");
}

/**
 * @brief init cluster_snaptho blossoms
 */
void
clusterCheckpointBlossoms()
{
    const std::string group = "Checkpoint";

    assert(HanamiRoot::root->addBlossom(group, "create", new CreateCheckpoint()));
    HanamiRoot::root->addEndpoint("v1/checkpoint",
                                  Kitsunemimi::Hanami::POST_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "create");

    assert(HanamiRoot::root->addBlossom(group, "finalize", new FinalizeCheckpoint()));
    HanamiRoot::root->addEndpoint("v1/checkpoint",
                                  Kitsunemimi::Hanami::PUT_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "finalize");

    assert(HanamiRoot::root->addBlossom(group, "get", new GetCheckpoint()));
    HanamiRoot::root->addEndpoint("v1/checkpoint",
                                  Kitsunemimi::Hanami::GET_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "get");

    assert(HanamiRoot::root->addBlossom(group, "delete", new DeleteCheckpoint()));
    HanamiRoot::root->addEndpoint("v1/checkpoint",
                                  Kitsunemimi::Hanami::DELETE_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "delete");

    assert(HanamiRoot::root->addBlossom(group, "list", new ListCheckpoint()));
    HanamiRoot::root->addEndpoint("v1/checkpoint/all",
                                  Kitsunemimi::Hanami::GET_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "list");
}

/**
 * @brief init request_result blossoms
 */
void
resultBlossoms()
{
    const std::string group = "Request-Result";

    assert(HanamiRoot::root->addBlossom(group, "get", new GetRequestResult()));
    HanamiRoot::root->addEndpoint("v1/request_result",
                           Kitsunemimi::Hanami::GET_TYPE,
                           BLOSSOM_TYPE,
                           group,
                           "get");

    assert(HanamiRoot::root->addBlossom(group, "list", new ListRequestResult()));
    HanamiRoot::root->addEndpoint("v1/request_result/all",
                                  Kitsunemimi::Hanami::GET_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "list");

    assert(HanamiRoot::root->addBlossom(group, "delete", new DeleteRequestResult()));
    HanamiRoot::root->addEndpoint("v1/request_result",
                                  Kitsunemimi::Hanami::DELETE_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "delete");
}

/**
 * @brief init logs blossoms
 */
void
logsBlossoms()
{
    const std::string group = "Logs";

    assert(HanamiRoot::root->addBlossom(group, "get_audit_log", new GetAuditLog()));
    HanamiRoot::root->addEndpoint("v1/audit_log",
                                  Kitsunemimi::Hanami::GET_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "get_audit_log");

    assert(HanamiRoot::root->addBlossom(group, "get_error_log", new GetErrorLog()));
    HanamiRoot::root->addEndpoint("v1/error_log",
                                  Kitsunemimi::Hanami::GET_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "get_error_log");
}

/**
 * @brief init token endpoints
 */
void
tokenBlossomes()
{
    const std::string group = "Token";

    assert(HanamiRoot::root->addBlossom(group, "create", new CreateToken()));
    HanamiRoot::root->addEndpoint("v1/token",
                                  Kitsunemimi::Hanami::POST_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "create");

    assert(HanamiRoot::root->addBlossom(group, "renew", new RenewToken()));
    HanamiRoot::root->addEndpoint("v1/token",
                                  Kitsunemimi::Hanami::PUT_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "renew");

    assert(HanamiRoot::root->addBlossom(group, "validate", new ValidateAccess()));
    HanamiRoot::root->addEndpoint("v1/auth",
                                  Kitsunemimi::Hanami::GET_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "validate");
}

/**
 * @brief documentation endpoints
 */
void
documentationBlossomes()
{
    const std::string group = "Documentation";

    assert(HanamiRoot::root->addBlossom(group, "generate_rest_api", new GenerateRestApiDocu()));
    HanamiRoot::root->addEndpoint("v1/documentation/api/rest",
                                  Kitsunemimi::Hanami::GET_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "generate_rest_api");
}

/**
 * @brief init user endpoints
 */
void
userBlossomes()
{
    const std::string userGroup = "User";

    assert(HanamiRoot::root->addBlossom(userGroup, "create", new CreateUser()));
    HanamiRoot::root->addEndpoint("v1/user",
                                  Kitsunemimi::Hanami::POST_TYPE,
                                  BLOSSOM_TYPE,
                                  userGroup,
                                  "create");

    assert(HanamiRoot::root->addBlossom(userGroup, "get", new GetUser()));
    HanamiRoot::root->addEndpoint("v1/user",
                                  Kitsunemimi::Hanami::GET_TYPE,
                                  BLOSSOM_TYPE,
                                  userGroup,
                                  "get");

    assert(HanamiRoot::root->addBlossom(userGroup, "list", new ListUsers()));
    HanamiRoot::root->addEndpoint("v1/user/all",
                                  Kitsunemimi::Hanami::GET_TYPE,
                                  BLOSSOM_TYPE,
                                  userGroup,
                                  "list");

    assert(HanamiRoot::root->addBlossom(userGroup, "delete", new DeleteUser()));
    HanamiRoot::root->addEndpoint("v1/user",
                                  Kitsunemimi::Hanami::DELETE_TYPE,
                                  BLOSSOM_TYPE,
                                  userGroup,
                                  "delete");

    const std::string userProjectGroup = "User-Projects";

    assert(HanamiRoot::root->addBlossom(userProjectGroup, "add_project", new AddProjectToUser()));
    HanamiRoot::root->addEndpoint("v1/user/project",
                                  Kitsunemimi::Hanami::POST_TYPE,
                                  BLOSSOM_TYPE,
                                  userProjectGroup,
                                  "add_project");

    assert(HanamiRoot::root->addBlossom(userProjectGroup, "remove_project", new RemoveProjectFromUser()));
    HanamiRoot::root->addEndpoint("v1/user/project",
                                  Kitsunemimi::Hanami::DELETE_TYPE,
                                  BLOSSOM_TYPE,
                                  userProjectGroup,
                                  "remove_project");

    // TODO: move ListUserProjects-class in user-directory
    assert(HanamiRoot::root->addBlossom(userProjectGroup, "list_user_projects", new ListUserProjects()));
    HanamiRoot::root->addEndpoint("v1/user/project",
                                  Kitsunemimi::Hanami::GET_TYPE,
                                  BLOSSOM_TYPE,
                                  userProjectGroup,
                                  "list_user_projects");
}

/**
 * @brief init project endpoints
 */
void
projectBlossomes()
{
    const std::string group = "Project";

    assert(HanamiRoot::root->addBlossom(group, "create", new CreateProject()));
    HanamiRoot::root->addEndpoint("v1/project",
                                  Kitsunemimi::Hanami::POST_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "create");

    assert(HanamiRoot::root->addBlossom(group, "get", new GetProject()));
    HanamiRoot::root->addEndpoint("v1/project",
                                  Kitsunemimi::Hanami::GET_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "get");

    assert(HanamiRoot::root->addBlossom(group, "list", new ListProjects()));
    HanamiRoot::root->addEndpoint("v1/project/all",
                                  Kitsunemimi::Hanami::GET_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "list");

    assert(HanamiRoot::root->addBlossom(group, "delete", new DeleteProject()));
    HanamiRoot::root->addEndpoint("v1/project",
                                  Kitsunemimi::Hanami::DELETE_TYPE,
                                  BLOSSOM_TYPE,
                                  group,
                                  "delete");
}

/**
 * @brief init measurement endpoints
 */
void
measuringBlossomes()
{
    const std::string group = "Measurements";

    assert(HanamiRoot::root->addBlossom(group, "get_power_consumption", new PowerConsumption()));
    assert(HanamiRoot::root->addEndpoint("v1/power_consumption",
                                         Kitsunemimi::Hanami::GET_TYPE,
                                         BLOSSOM_TYPE,
                                         group,
                                         "get_power_consumption"));

    assert(HanamiRoot::root->addBlossom(group, "get_speed", new Speed()));
    assert(HanamiRoot::root->addEndpoint("v1/speed",
                                         Kitsunemimi::Hanami::GET_TYPE,
                                         BLOSSOM_TYPE,
                                         group,
                                         "get_speed"));

    assert(HanamiRoot::root->addBlossom(group,
                                        "get_temperature_production",
                                        new ThermalProduction()));
    assert(HanamiRoot::root->addEndpoint("v1/temperature_production",
                                         Kitsunemimi::Hanami::GET_TYPE,
                                         BLOSSOM_TYPE,
                                         group,
                                         "get_temperature_production"));
}

/**
 * @brief initBlossoms
 */
void
initBlossoms()
{
    initClusterBlossoms();
    initTaskBlossoms();
    dataSetBlossoms();
    clusterCheckpointBlossoms();
    resultBlossoms();
    logsBlossoms();
    projectBlossomes();
    userBlossomes();
    documentationBlossomes();
    tokenBlossomes();
    measuringBlossomes();

    assert(HanamiRoot::root->addBlossom("System", "get_info", new GetSystemInfo()));
    assert(HanamiRoot::root->addEndpoint("v1/system_info",
                                         Kitsunemimi::Hanami::GET_TYPE,
                                         BLOSSOM_TYPE,
                                         "System",
                                         "get_info"));

    assert(HanamiRoot::root->addBlossom("Threading", "get_mapping", new GetThreadMapping()));
    assert(HanamiRoot::root->addEndpoint("v1/threading",
                                         Kitsunemimi::Hanami::GET_TYPE,
                                         BLOSSOM_TYPE,
                                         "Threading",
                                         "get_mapping"));

}

#endif // HANAMI_BLOSSOM_INITIALIZING_H
