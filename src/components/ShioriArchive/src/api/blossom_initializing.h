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

#ifndef SHIORIARCHIVE_BLOSSOM_INITIALIZING_H
#define SHIORIARCHIVE_BLOSSOM_INITIALIZING_H

#include <libKitsunemimiCommon/logger.h>

#include <libKitsunemimiHanamiNetwork/hanami_messaging.h>

#include <api/v1/data_files/list_data_set.h>
#include <api/v1/data_files/get_data_set.h>
#include <api/v1/data_files/delete_data_set.h>
#include <api/v1/data_files/check_data_set.h>
#include <api/v1/data_files/get_progress_data_set.h>
#include <api/v1/data_files/mnist/create_mnist_data_set.h>
#include <api/v1/data_files/mnist/finalize_mnist_data_set.h>
#include <api/v1/data_files/csv/create_csv_data_set.h>
#include <api/v1/data_files/csv/finalize_csv_data_set.h>

#include <api/v1/cluster_snapshot/create_cluster_snapshot.h>
#include <api/v1/cluster_snapshot/delete_cluster_snapshot.h>
#include <api/v1/cluster_snapshot/finish_cluster_snapshot.h>
#include <api/v1/cluster_snapshot/get_cluster_snapshot.h>
#include <api/v1/cluster_snapshot/list_cluster_snapshot.h>

#include <api/v1/request_results/delete_request_result.h>
#include <api/v1/request_results/get_request_result.h>
#include <api/v1/request_results/list_request_result.h>

#include <api/v1/logs/get_audit_log.h>
#include <api/v1/logs/get_error_log.h>

using Kitsunemimi::Hanami::HanamiMessaging;

/**
 * @brief init data_set blossoms
 */
void
dataSetBlossoms()
{
    HanamiMessaging* interface = HanamiMessaging::getInstance();
    const std::string group = "data_set";

    assert(interface->addBlossom(group, "create_mnist", new CreateMnistDataSet()));
    interface->addEndpoint("v1/mnist/data_set",
                           Kitsunemimi::Hanami::POST_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "create_mnist");

    assert(interface->addBlossom(group, "finalize_mnist", new FinalizeMnistDataSet()));
    interface->addEndpoint("v1/mnist/data_set",
                           Kitsunemimi::Hanami::PUT_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "finalize_mnist");

    assert(interface->addBlossom(group, "create_csv", new CreateCsvDataSet()));
    interface->addEndpoint("v1/csv/data_set",
                           Kitsunemimi::Hanami::POST_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "create_csv");

    assert(interface->addBlossom(group, "finalize_csv", new FinalizeCsvDataSet()));
    interface->addEndpoint("v1/csv/data_set",
                           Kitsunemimi::Hanami::PUT_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "finalize_csv");

    assert(interface->addBlossom(group, "check", new CheckDataSet()));
    interface->addEndpoint("v1/data_set/check",
                           Kitsunemimi::Hanami::POST_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "check");

    assert(interface->addBlossom(group, "progress", new GetProgressDataSet()));
    interface->addEndpoint("v1/data_set/progress",
                           Kitsunemimi::Hanami::GET_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "progress");

    assert(interface->addBlossom(group, "get", new GetDataSet()));
    interface->addEndpoint("v1/data_set",
                           Kitsunemimi::Hanami::GET_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "get");

    assert(interface->addBlossom(group, "delete", new DeleteDataSet()));
    interface->addEndpoint("v1/data_set",
                           Kitsunemimi::Hanami::DELETE_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "delete");

    assert(interface->addBlossom(group, "list", new ListDataSet()));
    interface->addEndpoint("v1/data_set/all",
                           Kitsunemimi::Hanami::GET_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "list");
}

/**
 * @brief init cluster_snaptho blossoms
 */
void
clusterSnapshotBlossoms()
{
    HanamiMessaging* interface = HanamiMessaging::getInstance();
    const std::string group = "cluster_snapshot";

    assert(interface->addBlossom(group, "create", new CreateClusterSnapshot()));
    interface->addEndpoint("v1/cluster_snapshot",
                           Kitsunemimi::Hanami::POST_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "create");

    assert(interface->addBlossom(group, "finalize", new FinalizeClusterSnapshot()));
    interface->addEndpoint("v1/cluster_snapshot",
                           Kitsunemimi::Hanami::PUT_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "finalize");

    assert(interface->addBlossom(group, "get", new GetClusterSnapshot()));
    interface->addEndpoint("v1/cluster_snapshot",
                           Kitsunemimi::Hanami::GET_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "get");

    assert(interface->addBlossom(group, "delete", new DeleteClusterSnapshot()));
    interface->addEndpoint("v1/cluster_snapshot",
                           Kitsunemimi::Hanami::DELETE_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "delete");

    assert(interface->addBlossom(group, "list", new ListClusterSnapshot()));
    interface->addEndpoint("v1/cluster_snapshot/all",
                           Kitsunemimi::Hanami::GET_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "list");
}

/**
 * @brief init request_result blossoms
 */
void
resultBlossoms()
{
    HanamiMessaging* interface = HanamiMessaging::getInstance();
    const std::string group = "request_result";

    assert(interface->addBlossom(group, "get", new GetRequestResult()));
    interface->addEndpoint("v1/request_result",
                           Kitsunemimi::Hanami::GET_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "get");

    assert(interface->addBlossom(group, "list", new ListRequestResult()));
    interface->addEndpoint("v1/request_result/all",
                           Kitsunemimi::Hanami::GET_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "list");

    assert(interface->addBlossom(group, "delete", new DeleteRequestResult()));
    interface->addEndpoint("v1/request_result",
                           Kitsunemimi::Hanami::DELETE_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "delete");
}

/**
 * @brief init logs blossoms
 */
void
logsBlossoms()
{
    HanamiMessaging* interface = HanamiMessaging::getInstance();
    const std::string group = "logs";

    assert(interface->addBlossom(group, "get_audit_log", new GetAuditLog()));
    interface->addEndpoint("v1/audit_log",
                           Kitsunemimi::Hanami::GET_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "get_audit_log");

    assert(interface->addBlossom(group, "get_error_log", new GetErrorLog()));
    interface->addEndpoint("v1/error_log",
                           Kitsunemimi::Hanami::GET_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "get_error_log");
}

void
initBlossoms()
{
    dataSetBlossoms();
    clusterSnapshotBlossoms();
    resultBlossoms();
    logsBlossoms();
}

#endif // SHIORIARCHIVE_BLOSSOM_INITIALIZING_H
