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
#include <api/http/v1/system_info/get_system_info.h>
#include <api/http/v1/task/create_request_task.h>
#include <api/http/v1/task/create_train_task.h>
#include <api/http/v1/task/delete_task.h>
#include <api/http/v1/task/list_task.h>
#include <api/http/v1/task/show_task.h>
#include <api/http/v1/threading/get_thread_mapping.h>
#include <api/http/v1/user/add_project_to_user.h>
#include <api/http/v1/user/create_user.h>
#include <api/http/v1/user/delete_user.h>
#include <api/http/v1/user/get_user.h>
#include <api/http/v1/user/list_user_projects.h>
#include <api/http/v1/user/list_users.h>
#include <api/http/v1/user/remove_project_from_user.h>
#include <hanami_common/files/text_file.h>
#include <hanami_common/functions/file_functions.h>
#include <hanami_common/logger.h>
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
    httpProcessing->addEndpoint("v1/cluster", POST_TYPE, group, new CreateCluster());
    httpProcessing->addEndpoint("v1/cluster", GET_TYPE, group, new ShowCluster());
    httpProcessing->addEndpoint("v1/cluster/all", GET_TYPE, group, new ListCluster());
    httpProcessing->addEndpoint("v1/cluster", DELETE_TYPE, group, new DeleteCluster());
    httpProcessing->addEndpoint("v1/cluster/save", POST_TYPE, group, new SaveCluster());
    httpProcessing->addEndpoint("v1/cluster/load", POST_TYPE, group, new LoadCluster());
    httpProcessing->addEndpoint("v1/cluster/set_mode", PUT_TYPE, group, new SetClusterMode());
    httpProcessing->addEndpoint("v1/cluster/switch_host", PUT_TYPE, group, new SwitchHosts());
}

/**
 * @brief initTaskBlossoms
 */
void
initTaskBlossoms()
{
    const std::string group = "Task";
    HttpProcessing* httpProcessing = HanamiRoot::httpServer->httpProcessing;
    httpProcessing->addEndpoint("v1/task/train", POST_TYPE, group, new CreateTrainTask());
    httpProcessing->addEndpoint("v1/task/request", POST_TYPE, group, new CreateRequestTask());
    httpProcessing->addEndpoint("v1/task", GET_TYPE, group, new ShowTask());
    httpProcessing->addEndpoint("v1/task/all", GET_TYPE, group, new ListTask());
    httpProcessing->addEndpoint("v1/task", DELETE_TYPE, group, new DeleteTask());
}

/**
 * @brief init dataset blossoms
 */
void
datasetBlossoms()
{
    const std::string group = "Data Set";
    HttpProcessing* httpProcessing = HanamiRoot::httpServer->httpProcessing;
    httpProcessing->addEndpoint(
        "v1/dataset/upload/mnist", POST_TYPE, group, new CreateMnistDataSet());
    httpProcessing->addEndpoint(
        "v1/dataset/upload/mnist", PUT_TYPE, group, new FinalizeMnistDataSet());
    httpProcessing->addEndpoint("v1/dataset/upload/csv", POST_TYPE, group, new CreateCsvDataSet());
    httpProcessing->addEndpoint("v1/dataset/upload/csv", PUT_TYPE, group, new FinalizeCsvDataSet());
    httpProcessing->addEndpoint("v1/dataset/check", GET_TYPE, group, new CheckMnistDataSet());
    httpProcessing->addEndpoint("v1/dataset/progress", GET_TYPE, group, new GetProgressDataSet());
    httpProcessing->addEndpoint("v1/dataset", GET_TYPE, group, new GetDataSet());
    httpProcessing->addEndpoint("v1/dataset", DELETE_TYPE, group, new DeleteDataSet());
    httpProcessing->addEndpoint("v1/dataset/all", GET_TYPE, group, new ListDataSet());
}

/**
 * @brief init cluster_snaptho blossoms
 */
void
clusterCheckpointBlossoms()
{
    const std::string group = "Checkpoint";
    HttpProcessing* httpProcessing = HanamiRoot::httpServer->httpProcessing;
    httpProcessing->addEndpoint("v1/checkpoint", GET_TYPE, group, new GetCheckpoint());
    httpProcessing->addEndpoint("v1/checkpoint", DELETE_TYPE, group, new DeleteCheckpoint());
    httpProcessing->addEndpoint("v1/checkpoint/all", GET_TYPE, group, new ListCheckpoint());
}

/**
 * @brief init logs blossoms
 */
void
logsBlossoms()
{
    const std::string group = "Logs";
    HttpProcessing* httpProcessing = HanamiRoot::httpServer->httpProcessing;
    httpProcessing->addEndpoint("v1/audit_log", GET_TYPE, group, new GetAuditLog());
    httpProcessing->addEndpoint("v1/error_log", GET_TYPE, group, new GetErrorLog());
}

/**
 * @brief init token endpoints
 */
void
tokenBlossomes()
{
    const std::string group = "Token";
    HttpProcessing* httpProcessing = HanamiRoot::httpServer->httpProcessing;
    httpProcessing->addEndpoint("v1/token", POST_TYPE, group, new CreateToken());
    httpProcessing->addEndpoint("v1/token", PUT_TYPE, group, new RenewToken());
    httpProcessing->addEndpoint("v1/auth", GET_TYPE, group, new ValidateAccess());
}

/**
 * @brief init user endpoints
 */
void
userBlossomes()
{
    const std::string group = "User";
    HttpProcessing* httpProcessing = HanamiRoot::httpServer->httpProcessing;
    httpProcessing->addEndpoint("v1/user", POST_TYPE, group, new CreateUser());
    httpProcessing->addEndpoint("v1/user", GET_TYPE, group, new GetUser());
    httpProcessing->addEndpoint("v1/user/all", GET_TYPE, group, new ListUsers());
    httpProcessing->addEndpoint("v1/user", DELETE_TYPE, group, new DeleteUser());

    httpProcessing->addEndpoint("v1/user/project", POST_TYPE, group, new AddProjectToUser());
    httpProcessing->addEndpoint("v1/user/project", DELETE_TYPE, group, new RemoveProjectFromUser());
    httpProcessing->addEndpoint("v1/user/project", GET_TYPE, group, new ListUserProjects());
}

/**
 * @brief init project endpoints
 */
void
projectBlossomes()
{
    const std::string group = "Project";
    HttpProcessing* httpProcessing = HanamiRoot::httpServer->httpProcessing;
    httpProcessing->addEndpoint("v1/project", POST_TYPE, group, new CreateProject());
    httpProcessing->addEndpoint("v1/project", GET_TYPE, group, new GetProject());
    httpProcessing->addEndpoint("v1/project/all", GET_TYPE, group, new ListProjects());
    httpProcessing->addEndpoint("v1/project", DELETE_TYPE, group, new DeleteProject());
}

/**
 * @brief init measurement endpoints
 */
void
measuringBlossomes()
{
    const std::string group = "Measurements";
    HttpProcessing* httpProcessing = HanamiRoot::httpServer->httpProcessing;
    httpProcessing->addEndpoint("v1/power_consumption", GET_TYPE, group, new PowerConsumption());
    httpProcessing->addEndpoint("v1/speed", GET_TYPE, group, new Speed());
    httpProcessing->addEndpoint(
        "v1/temperature_production", GET_TYPE, group, new ThermalProduction());
}

/**
 * @brief init host endpoints
 */
void
hostBlossoms()
{
    const std::string group = "Hosts";
    HttpProcessing* httpProcessing = HanamiRoot::httpServer->httpProcessing;
    httpProcessing->addEndpoint("v1/host/all", GET_TYPE, group, new ListHosts());
}

/**
 * @brief initBlossoms
 */
void
initBlossoms()
{
    clusterCheckpointBlossoms();
    datasetBlossoms();
    hostBlossoms();
    initClusterBlossoms();
    initTaskBlossoms();
    logsBlossoms();
    measuringBlossomes();
    projectBlossomes();
    tokenBlossomes();
    userBlossomes();

    HttpProcessing* httpProcessing = HanamiRoot::httpServer->httpProcessing;
    httpProcessing->addEndpoint("v1/system_info", GET_TYPE, "System", new GetSystemInfo());
    httpProcessing->addEndpoint("v1/threading", GET_TYPE, "Threading", new GetThreadMapping());
}

#endif  // HANAMI_BLOSSOM_INITIALIZING_H
