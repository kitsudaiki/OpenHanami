/**
 * @file        endpoint_init.cpp
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

#include "endpoint_init.h"

void
initV1M0alphaEndpoints()
{
    HttpProcessing* httpProcessing = HanamiRoot::httpServer->httpProcessing;
    const std::string version = "v1.0alpha";
    std::string group = "";

    group = "Cluster";
    httpProcessing->addEndpoint(version + "/cluster", POST_TYPE, group, new CreateClusterV1M0());
    httpProcessing->addEndpoint(version + "/cluster", GET_TYPE, group, new ShowClusterV1M0());
    httpProcessing->addEndpoint(version + "/cluster/all", GET_TYPE, group, new ListClusterV1M0());
    httpProcessing->addEndpoint(version + "/cluster", DELETE_TYPE, group, new DeleteClusterV1M0());
    httpProcessing->addEndpoint(version + "/cluster/save", POST_TYPE, group, new SaveClusterV1M0());
    httpProcessing->addEndpoint(version + "/cluster/load", POST_TYPE, group, new LoadClusterV1M0());
    httpProcessing->addEndpoint(
        version + "/cluster/set_mode", PUT_TYPE, group, new SetClusterModeV1M0());
    httpProcessing->addEndpoint(
        version + "/cluster/switch_host", PUT_TYPE, group, new SwitchHostsV1M0());

    group = "Task";
    httpProcessing->addEndpoint(
        version + "/task/train", POST_TYPE, group, new CreateTrainTaskV1M0());
    httpProcessing->addEndpoint(
        version + "/task/request", POST_TYPE, group, new CreateRequestTaskV1M0());
    httpProcessing->addEndpoint(version + "/task", GET_TYPE, group, new ShowTaskV1M0());
    httpProcessing->addEndpoint(version + "/task/all", GET_TYPE, group, new ListTaskV1M0());
    httpProcessing->addEndpoint(version + "/task", DELETE_TYPE, group, new DeleteTaskV1M0());

    group = "Data Set";
    httpProcessing->addEndpoint(
        version + "/dataset/upload/mnist", POST_TYPE, group, new CreateMnistDataSetV1M0());
    httpProcessing->addEndpoint(
        version + "/dataset/upload/mnist", PUT_TYPE, group, new FinalizeMnistDataSetV1M0());
    httpProcessing->addEndpoint(
        version + "/dataset/upload/csv", POST_TYPE, group, new CreateCsvDataSetV1M0());
    httpProcessing->addEndpoint(
        version + "/dataset/upload/csv", PUT_TYPE, group, new FinalizeCsvDataSetV1M0());
    httpProcessing->addEndpoint(
        version + "/dataset/check", GET_TYPE, group, new CheckMnistDataSetV1M0());
    httpProcessing->addEndpoint(
        version + "/dataset/progress", GET_TYPE, group, new GetProgressDataSetV1M0());
    httpProcessing->addEndpoint(version + "/dataset", GET_TYPE, group, new GetDataSetV1M0());
    httpProcessing->addEndpoint(version + "/dataset", DELETE_TYPE, group, new DeleteDataSetV1M0());
    httpProcessing->addEndpoint(version + "/dataset/all", GET_TYPE, group, new ListDataSetV1M0());

    group = "Checkpoint";
    httpProcessing->addEndpoint(version + "/checkpoint", GET_TYPE, group, new GetCheckpointV1M0());
    httpProcessing->addEndpoint(
        version + "/checkpoint", DELETE_TYPE, group, new DeleteCheckpointV1M0());
    httpProcessing->addEndpoint(
        version + "/checkpoint/all", GET_TYPE, group, new ListCheckpointV1M0());

    group = "Logs";
    httpProcessing->addEndpoint(version + "/audit_log", GET_TYPE, group, new GetAuditLogV1M0());
    httpProcessing->addEndpoint(version + "/error_log", GET_TYPE, group, new GetErrorLogV1M0());

    group = "Token";
    httpProcessing->addEndpoint(version + "/token", POST_TYPE, group, new CreateTokenV1M0());
    httpProcessing->addEndpoint(version + "/token", PUT_TYPE, group, new RenewTokenV1M0());
    httpProcessing->addEndpoint(version + "/auth", GET_TYPE, group, new ValidateAccessV1M0());

    group = "User";
    httpProcessing->addEndpoint(version + "/user", POST_TYPE, group, new CreateUserV1M0());
    httpProcessing->addEndpoint(version + "/user", GET_TYPE, group, new GetUserV1M0());
    httpProcessing->addEndpoint(version + "/user/all", GET_TYPE, group, new ListUsersV1M0());
    httpProcessing->addEndpoint(version + "/user", DELETE_TYPE, group, new DeleteUserV1M0());

    httpProcessing->addEndpoint(
        version + "/user/project", POST_TYPE, group, new AddProjectToUserV1M0());
    httpProcessing->addEndpoint(
        version + "/user/project", DELETE_TYPE, group, new RemoveProjectFromUserV1M0());
    httpProcessing->addEndpoint(
        version + "/user/project", GET_TYPE, group, new ListUserProjectsV1M0());

    group = "Project";
    httpProcessing->addEndpoint(version + "/project", POST_TYPE, group, new CreateProjectV1M0());
    httpProcessing->addEndpoint(version + "/project", GET_TYPE, group, new GetProjectV1M0());
    httpProcessing->addEndpoint(version + "/project/all", GET_TYPE, group, new ListProjectsV1M0());
    httpProcessing->addEndpoint(version + "/project", DELETE_TYPE, group, new DeleteProjectV1M0());

    httpProcessing->addEndpoint(version + "/host/all", GET_TYPE, group, new ListHostsV1M0());

    group = "Measurements";
    httpProcessing->addEndpoint(
        version + "/power_consumption", GET_TYPE, group, new PowerConsumptionV1M0());
    httpProcessing->addEndpoint(version + "/speed", GET_TYPE, group, new SpeedV1M0());
    httpProcessing->addEndpoint(
        version + "/temperature_production", GET_TYPE, group, new ThermalProductionV1M0());

    httpProcessing->addEndpoint(
        version + "/system_info", GET_TYPE, "System", new GetSystemInfoV1M0());
    httpProcessing->addEndpoint(
        version + "/threading", GET_TYPE, "Threading", new GetThreadMappingV1M0());
}

void
initEndpoints()
{
    initV1M0alphaEndpoints();
}
