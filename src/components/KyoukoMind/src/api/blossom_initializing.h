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

#ifndef KYOUKOMIND_BLOSSOM_INITIALIZING_H
#define KYOUKOMIND_BLOSSOM_INITIALIZING_H

#include <common.h>

#include <libKitsunemimiConfig/config_handler.h>
#include <libKitsunemimiCommon/methods/file_methods.h>
#include <libKitsunemimiCommon/files/text_file.h>
#include <libKitsunemimiCommon/logger.h>

#include <libKitsunemimiHanamiNetwork/hanami_messaging.h>

#include <api/v1/cluster/create_cluster.h>
#include <api/v1/cluster/show_cluster.h>
#include <api/v1/cluster/list_cluster.h>
#include <api/v1/cluster/delete_cluster.h>
#include <api/v1/cluster/save_cluster.h>
#include <api/v1/cluster/load_cluster.h>
#include <api/v1/cluster/set_cluster_mode.h>

#include <api/v1/template/upload_template.h>
#include <api/v1/template/delete_template.h>
#include <api/v1/template/list_templates.h>
#include <api/v1/template/show_template.h>

#include <api/v1/task/create_task.h>
#include <api/v1/task/show_task.h>
#include <api/v1/task/list_task.h>
#include <api/v1/task/delete_task.h>

using Kitsunemimi::Hanami::HanamiMessaging;

/**
 * @brief initClusterBlossoms
 */
void
initClusterBlossoms()
{
    HanamiMessaging* interface = HanamiMessaging::getInstance();
    const std::string group = "cluster";

    assert(interface->addBlossom(group, "create", new CreateCluster()));
    interface->addEndpoint("v1/cluster",
                           Kitsunemimi::Hanami::POST_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "create");

    assert(interface->addBlossom(group, "show", new ShowCluster()));
    interface->addEndpoint("v1/cluster",
                           Kitsunemimi::Hanami::GET_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "show");

    assert(interface->addBlossom(group, "list", new ListCluster()));
    interface->addEndpoint("v1/cluster/all",
                           Kitsunemimi::Hanami::GET_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "list");

    assert(interface->addBlossom(group, "delete", new DeleteCluster()));
    interface->addEndpoint("v1/cluster",
                           Kitsunemimi::Hanami::DELETE_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "delete");

    assert(interface->addBlossom(group, "save", new SaveCluster()));
    interface->addEndpoint("v1/cluster/save",
                           Kitsunemimi::Hanami::POST_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "save");

    assert(interface->addBlossom(group, "load", new LoadCluster()));
    interface->addEndpoint("v1/cluster/load",
                           Kitsunemimi::Hanami::POST_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "load");

    assert(interface->addBlossom(group, "set_mode", new SetClusterMode()));
    interface->addEndpoint("v1/cluster/set_mode",
                           Kitsunemimi::Hanami::PUT_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "set_mode");
}

/**
 * @brief initTemplateBlossoms
 */
void
initTemplateBlossoms()
{
    HanamiMessaging* interface = HanamiMessaging::getInstance();
    const std::string group = "template";

    assert(interface->addBlossom(group, "upload", new UploadTemplate()));
    interface->addEndpoint("v1/template/upload",
                           Kitsunemimi::Hanami::POST_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "upload");

    assert(interface->addBlossom(group, "show", new ShowTemplate()));
    interface->addEndpoint("v1/template",
                           Kitsunemimi::Hanami::GET_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "show");

    assert(interface->addBlossom(group, "list", new ListTemplates()));
    interface->addEndpoint("v1/template/all",
                           Kitsunemimi::Hanami::GET_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "list");

    assert(interface->addBlossom(group, "delete", new DeleteTemplate()));
    interface->addEndpoint("v1/template",
                           Kitsunemimi::Hanami::DELETE_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "delete");
}

/**
 * @brief initTaskBlossoms
 */
void
initTaskBlossoms()
{
    HanamiMessaging* interface = HanamiMessaging::getInstance();
    const std::string group = "task";

    assert(interface->addBlossom(group, "create", new CreateTask()));
    interface->addEndpoint("v1/task",
                           Kitsunemimi::Hanami::POST_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "create");

    assert(interface->addBlossom(group, "show", new ShowTask()));
    interface->addEndpoint("v1/task",
                           Kitsunemimi::Hanami::GET_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "show");

    assert(interface->addBlossom(group, "list", new ListTask()));
    interface->addEndpoint("v1/task/all",
                           Kitsunemimi::Hanami::GET_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "list");

    assert(interface->addBlossom(group, "delete", new DeleteTask()));
    interface->addEndpoint("v1/task",
                           Kitsunemimi::Hanami::DELETE_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "delete");
}

/**
 * @brief initBlossoms
 */
void
initBlossoms()
{
    initClusterBlossoms();
    initTemplateBlossoms();
    initTaskBlossoms();
}

#endif // KYOUKOMIND_BLOSSOM_INITIALIZING_H
