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

#include <api/endpoint_processing/http_processing/http_processing.h>
#include <api/endpoint_processing/http_server.h>
#include <api/http/v1/auth/create_token.h>
#include <api/http/v1/auth/list_user_projects.h>
#include <api/http/v1/auth/renew_token.h>
#include <api/http/v1/auth/validate_access.h>
#include <api/http/v1/checkpoint/delete_checkpoint.h>
#include <api/http/v1/checkpoint/get_checkpoint.h>
#include <api/http/v1/checkpoint/list_checkpoint.h>
#include <api/http/v1/cluster/create_cluster.h>
#include <api/http/v1/cluster/delete_cluster.h>
#include <api/http/v1/cluster/list_cluster.h>
#include <api/http/v1/cluster/load_cluster.h>
#include <api/http/v1/cluster/save_cluster.h>
#include <api/http/v1/cluster/set_cluster_mode.h>
#include <api/http/v1/cluster/show_cluster.h>
#include <api/http/v1/cluster/switch_hosts.h>
#include <api/http/v1/data_files/check_dataset.h>
#include <api/http/v1/data_files/csv/create_csv_dataset.h>
#include <api/http/v1/data_files/csv/finalize_csv_dataset.h>
#include <api/http/v1/data_files/delete_dataset.h>
#include <api/http/v1/data_files/get_dataset.h>
#include <api/http/v1/data_files/get_progress_dataset.h>
#include <api/http/v1/data_files/list_dataset.h>
#include <api/http/v1/data_files/mnist/create_mnist_dataset.h>
#include <api/http/v1/data_files/mnist/finalize_mnist_dataset.h>
#include <api/http/v1/hosts/list_hosts.h>
#include <api/http/v1/logs/get_audit_log.h>
#include <api/http/v1/logs/get_error_log.h>
#include <api/http/v1/measurements/power_consumption.h>
#include <api/http/v1/measurements/speed.h>
#include <api/http/v1/measurements/temperature_production.h>
#include <api/http/v1/project/create_project.h>
#include <api/http/v1/project/delete_project.h>
#include <api/http/v1/project/get_project.h>
#include <api/http/v1/project/list_projects.h>
#include <api/http/v1/request_results/delete_request_result.h>
#include <api/http/v1/request_results/get_request_result.h>
#include <api/http/v1/request_results/list_request_result.h>
#include <api/http/v1/system_info/get_system_info.h>
#include <api/http/v1/task/create_task.h>
#include <api/http/v1/task/delete_task.h>
#include <api/http/v1/task/list_task.h>
#include <api/http/v1/task/show_task.h>
#include <api/http/v1/threading/get_thread_mapping.h>
#include <api/http/v1/user/add_project_to_user.h>
#include <api/http/v1/user/create_user.h>
#include <api/http/v1/user/delete_user.h>
#include <api/http/v1/user/get_user.h>
#include <api/http/v1/user/list_users.h>
#include <api/http/v1/user/remove_project_from_user.h>
#include <common.h>
#include <hanami_common/files/text_file.h>
#include <hanami_common/logger.h>
#include <hanami_common/methods/file_methods.h>
#include <hanami_config/config_handler.h>
#include <hanami_root.h>

/**
 * @brief initClusterBlossoms
 */
void
initClusterBlossoms()
{
    const std::string group = "Cluster";
    HttpProcessing* httpProcessing = HanamiRoot::httpServer->httpProcessing;

    httpProcessing->addBlossom(group, "create", new CreateCluster());
    httpProcessing->addEndpoint("v1/cluster", Hanami::POST_TYPE, BLOSSOM_TYPE, group, "create");

    httpProcessing->addBlossom(group, "show", new ShowCluster());
    httpProcessing->addEndpoint("v1/cluster", Hanami::GET_TYPE, BLOSSOM_TYPE, group, "show");

    httpProcessing->addBlossom(group, "list", new ListCluster());
    httpProcessing->addEndpoint("v1/cluster/all", Hanami::GET_TYPE, BLOSSOM_TYPE, group, "list");

    httpProcessing->addBlossom(group, "delete", new DeleteCluster());
    httpProcessing->addEndpoint("v1/cluster", Hanami::DELETE_TYPE, BLOSSOM_TYPE, group, "delete");

    httpProcessing->addBlossom(group, "save", new SaveCluster());
    httpProcessing->addEndpoint("v1/cluster/save", Hanami::POST_TYPE, BLOSSOM_TYPE, group, "save");

    httpProcessing->addBlossom(group, "load", new LoadCluster());
    httpProcessing->addEndpoint("v1/cluster/load", Hanami::POST_TYPE, BLOSSOM_TYPE, group, "load");

    httpProcessing->addBlossom(group, "set_mode", new SetClusterMode());
    httpProcessing->addEndpoint(
        "v1/cluster/set_mode", Hanami::PUT_TYPE, BLOSSOM_TYPE, group, "set_mode");

    httpProcessing->addBlossom(group, "switch_host", new SwitchHosts());
    httpProcessing->addEndpoint(
        "v1/cluster/switch_host", Hanami::PUT_TYPE, BLOSSOM_TYPE, group, "switch_host");
}

/**
 * @brief initTaskBlossoms
 */
void
initTaskBlossoms()
{
    const std::string group = "Task";
    HttpProcessing* httpProcessing = HanamiRoot::httpServer->httpProcessing;

    httpProcessing->addBlossom(group, "create", new CreateTask());
    httpProcessing->addEndpoint("v1/task", Hanami::POST_TYPE, BLOSSOM_TYPE, group, "create");

    httpProcessing->addBlossom(group, "show", new ShowTask());
    httpProcessing->addEndpoint("v1/task", Hanami::GET_TYPE, BLOSSOM_TYPE, group, "show");

    httpProcessing->addBlossom(group, "list", new ListTask());
    httpProcessing->addEndpoint("v1/task/all", Hanami::GET_TYPE, BLOSSOM_TYPE, group, "list");

    httpProcessing->addBlossom(group, "delete", new DeleteTask());
    httpProcessing->addEndpoint("v1/task", Hanami::DELETE_TYPE, BLOSSOM_TYPE, group, "delete");
}

/**
 * @brief init dataset blossoms
 */
void
dataSetBlossoms()
{
    const std::string group = "Data Set";
    HttpProcessing* httpProcessing = HanamiRoot::httpServer->httpProcessing;

    httpProcessing->addBlossom(group, "create_mnist", new CreateMnistDataSet());
    httpProcessing->addEndpoint(
        "v1/mnist/dataset", Hanami::POST_TYPE, BLOSSOM_TYPE, group, "create_mnist");

    httpProcessing->addBlossom(group, "finalize_mnist", new FinalizeMnistDataSet());
    httpProcessing->addEndpoint(
        "v1/mnist/dataset", Hanami::PUT_TYPE, BLOSSOM_TYPE, group, "finalize_mnist");

    httpProcessing->addBlossom(group, "create_csv", new CreateCsvDataSet());
    httpProcessing->addEndpoint(
        "v1/csv/dataset", Hanami::POST_TYPE, BLOSSOM_TYPE, group, "create_csv");

    httpProcessing->addBlossom(group, "finalize_csv", new FinalizeCsvDataSet());
    httpProcessing->addEndpoint(
        "v1/csv/dataset", Hanami::PUT_TYPE, BLOSSOM_TYPE, group, "finalize_csv");

    httpProcessing->addBlossom(group, "check", new CheckDataSet());
    httpProcessing->addEndpoint(
        "v1/dataset/check", Hanami::POST_TYPE, BLOSSOM_TYPE, group, "check");

    httpProcessing->addBlossom(group, "progress", new GetProgressDataSet());
    httpProcessing->addEndpoint(
        "v1/dataset/progress", Hanami::GET_TYPE, BLOSSOM_TYPE, group, "progress");

    httpProcessing->addBlossom(group, "get", new GetDataSet());
    httpProcessing->addEndpoint("v1/dataset", Hanami::GET_TYPE, BLOSSOM_TYPE, group, "get");

    httpProcessing->addBlossom(group, "delete", new DeleteDataSet());
    httpProcessing->addEndpoint("v1/dataset", Hanami::DELETE_TYPE, BLOSSOM_TYPE, group, "delete");

    httpProcessing->addBlossom(group, "list", new ListDataSet());
    httpProcessing->addEndpoint("v1/dataset/all", Hanami::GET_TYPE, BLOSSOM_TYPE, group, "list");
}

/**
 * @brief init cluster_snaptho blossoms
 */
void
clusterCheckpointBlossoms()
{
    const std::string group = "Checkpoint";
    HttpProcessing* httpProcessing = HanamiRoot::httpServer->httpProcessing;

    httpProcessing->addBlossom(group, "get", new GetCheckpoint());
    httpProcessing->addEndpoint("v1/checkpoint", Hanami::GET_TYPE, BLOSSOM_TYPE, group, "get");

    httpProcessing->addBlossom(group, "delete", new DeleteCheckpoint());
    httpProcessing->addEndpoint(
        "v1/checkpoint", Hanami::DELETE_TYPE, BLOSSOM_TYPE, group, "delete");

    httpProcessing->addBlossom(group, "list", new ListCheckpoint());
    httpProcessing->addEndpoint("v1/checkpoint/all", Hanami::GET_TYPE, BLOSSOM_TYPE, group, "list");
}

/**
 * @brief init request_result blossoms
 */
void
resultBlossoms()
{
    const std::string group = "Request-Result";
    HttpProcessing* httpProcessing = HanamiRoot::httpServer->httpProcessing;

    httpProcessing->addBlossom(group, "get", new GetRequestResult());
    httpProcessing->addEndpoint("v1/request_result", Hanami::GET_TYPE, BLOSSOM_TYPE, group, "get");

    httpProcessing->addBlossom(group, "list", new ListRequestResult());
    httpProcessing->addEndpoint(
        "v1/request_result/all", Hanami::GET_TYPE, BLOSSOM_TYPE, group, "list");

    httpProcessing->addBlossom(group, "delete", new DeleteRequestResult());
    httpProcessing->addEndpoint(
        "v1/request_result", Hanami::DELETE_TYPE, BLOSSOM_TYPE, group, "delete");
}

/**
 * @brief init logs blossoms
 */
void
logsBlossoms()
{
    const std::string group = "Logs";
    HttpProcessing* httpProcessing = HanamiRoot::httpServer->httpProcessing;

    httpProcessing->addBlossom(group, "get_audit_log", new GetAuditLog());
    httpProcessing->addEndpoint(
        "v1/audit_log", Hanami::GET_TYPE, BLOSSOM_TYPE, group, "get_audit_log");

    httpProcessing->addBlossom(group, "get_error_log", new GetErrorLog());
    httpProcessing->addEndpoint(
        "v1/error_log", Hanami::GET_TYPE, BLOSSOM_TYPE, group, "get_error_log");
}

/**
 * @brief init token endpoints
 */
void
tokenBlossomes()
{
    const std::string group = "Token";
    HttpProcessing* httpProcessing = HanamiRoot::httpServer->httpProcessing;

    httpProcessing->addBlossom(group, "create", new CreateToken());
    httpProcessing->addEndpoint("v1/token", Hanami::POST_TYPE, BLOSSOM_TYPE, group, "create");

    httpProcessing->addBlossom(group, "renew", new RenewToken());
    httpProcessing->addEndpoint("v1/token", Hanami::PUT_TYPE, BLOSSOM_TYPE, group, "renew");

    httpProcessing->addBlossom(group, "validate", new ValidateAccess());
    httpProcessing->addEndpoint("v1/auth", Hanami::GET_TYPE, BLOSSOM_TYPE, group, "validate");
}

/**
 * @brief init user endpoints
 */
void
userBlossomes()
{
    const std::string userGroup = "User";
    HttpProcessing* httpProcessing = HanamiRoot::httpServer->httpProcessing;

    httpProcessing->addBlossom(userGroup, "create", new CreateUser());
    httpProcessing->addEndpoint("v1/user", Hanami::POST_TYPE, BLOSSOM_TYPE, userGroup, "create");

    httpProcessing->addBlossom(userGroup, "get", new GetUser());
    httpProcessing->addEndpoint("v1/user", Hanami::GET_TYPE, BLOSSOM_TYPE, userGroup, "get");

    httpProcessing->addBlossom(userGroup, "list", new ListUsers());
    httpProcessing->addEndpoint("v1/user/all", Hanami::GET_TYPE, BLOSSOM_TYPE, userGroup, "list");

    httpProcessing->addBlossom(userGroup, "delete", new DeleteUser());
    httpProcessing->addEndpoint("v1/user", Hanami::DELETE_TYPE, BLOSSOM_TYPE, userGroup, "delete");

    const std::string userProjectGroup = "User-Projects";

    httpProcessing->addBlossom(userProjectGroup, "add_project", new AddProjectToUser());
    httpProcessing->addEndpoint(
        "v1/user/project", Hanami::POST_TYPE, BLOSSOM_TYPE, userProjectGroup, "add_project");

    httpProcessing->addBlossom(userProjectGroup, "remove_project", new RemoveProjectFromUser());
    httpProcessing->addEndpoint(
        "v1/user/project", Hanami::DELETE_TYPE, BLOSSOM_TYPE, userProjectGroup, "remove_project");

    // TODO: move ListUserProjects-class in user-directory

    httpProcessing->addBlossom(userProjectGroup, "list_user_projects", new ListUserProjects());
    httpProcessing->addEndpoint(
        "v1/user/project", Hanami::GET_TYPE, BLOSSOM_TYPE, userProjectGroup, "list_user_projects");
}

/**
 * @brief init project endpoints
 */
void
projectBlossomes()
{
    const std::string group = "Project";
    HttpProcessing* httpProcessing = HanamiRoot::httpServer->httpProcessing;

    httpProcessing->addBlossom(group, "create", new CreateProject());
    httpProcessing->addEndpoint("v1/project", Hanami::POST_TYPE, BLOSSOM_TYPE, group, "create");

    httpProcessing->addBlossom(group, "get", new GetProject());
    httpProcessing->addEndpoint("v1/project", Hanami::GET_TYPE, BLOSSOM_TYPE, group, "get");

    httpProcessing->addBlossom(group, "list", new ListProjects());
    httpProcessing->addEndpoint("v1/project/all", Hanami::GET_TYPE, BLOSSOM_TYPE, group, "list");

    httpProcessing->addBlossom(group, "delete", new DeleteProject());
    httpProcessing->addEndpoint("v1/project", Hanami::DELETE_TYPE, BLOSSOM_TYPE, group, "delete");
}

/**
 * @brief init measurement endpoints
 */
void
measuringBlossomes()
{
    const std::string group = "Measurements";
    HttpProcessing* httpProcessing = HanamiRoot::httpServer->httpProcessing;

    httpProcessing->addBlossom(group, "get_power_consumption", new PowerConsumption());
    httpProcessing->addEndpoint(
        "v1/power_consumption", Hanami::GET_TYPE, BLOSSOM_TYPE, group, "get_power_consumption");

    httpProcessing->addBlossom(group, "get_speed", new Speed());
    httpProcessing->addEndpoint("v1/speed", Hanami::GET_TYPE, BLOSSOM_TYPE, group, "get_speed");

    httpProcessing->addBlossom(group, "get_temperature_production", new ThermalProduction());
    httpProcessing->addEndpoint("v1/temperature_production",
                                Hanami::GET_TYPE,
                                BLOSSOM_TYPE,
                                group,
                                "get_temperature_production");
}

/**
 * @brief init host endpoints
 */
void
hostBlossoms()
{
    const std::string group = "Hosts";
    HttpProcessing* httpProcessing = HanamiRoot::httpServer->httpProcessing;

    httpProcessing->addBlossom(group, "list_hosts", new ListHosts());
    httpProcessing->addEndpoint("v1/host/all", Hanami::GET_TYPE, BLOSSOM_TYPE, group, "list_hosts");
}

/**
 * @brief initBlossoms
 */
void
initBlossoms()
{
    clusterCheckpointBlossoms();
    dataSetBlossoms();
    hostBlossoms();
    initClusterBlossoms();
    initTaskBlossoms();
    logsBlossoms();
    measuringBlossomes();
    projectBlossomes();
    resultBlossoms();
    tokenBlossomes();
    userBlossomes();

    HttpProcessing* httpProcessing = HanamiRoot::httpServer->httpProcessing;

    httpProcessing->addBlossom("System", "get_info", new GetSystemInfo());
    httpProcessing->addEndpoint(
        "v1/system_info", Hanami::GET_TYPE, BLOSSOM_TYPE, "System", "get_info");

    httpProcessing->addBlossom("Threading", "get_mapping", new GetThreadMapping());
    httpProcessing->addEndpoint(
        "v1/threading", Hanami::GET_TYPE, BLOSSOM_TYPE, "Threading", "get_mapping");
}

#endif  // HANAMI_BLOSSOM_INITIALIZING_H
