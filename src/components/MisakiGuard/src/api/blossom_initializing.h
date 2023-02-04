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

#ifndef MISAKIGUARD_BLOSSOM_INITIALIZING_H
#define MISAKIGUARD_BLOSSOM_INITIALIZING_H

#include <libKitsunemimiCommon/logger.h>

#include <libKitsunemimiHanamiNetwork/hanami_messaging.h>

#include <api/v1/user/create_user.h>
#include <api/v1/user/get_user.h>
#include <api/v1/user/list_users.h>
#include <api/v1/user/delete_user.h>
#include <api/v1/user/add_project_to_user.h>
#include <api/v1/user/remove_project_from_user.h>

#include <api/v1/project/create_project.h>
#include <api/v1/project/get_project.h>
#include <api/v1/project/list_projects.h>
#include <api/v1/project/delete_project.h>

#include  <api/v1/documentation/generate_rest_api_docu.h>

#include <api/v1/auth/create_internal_token.h>
#include <api/v1/auth/create_token.h>
#include <api/v1/auth/validate_access.h>
#include <api/v1/auth/list_user_projects.h>
#include <api/v1/auth/renew_token.h>

using Kitsunemimi::Hanami::HanamiMessaging;

/**
 * @brief init token endpoints
 */
void
tokenBlossomes()
{
    HanamiMessaging* interface = HanamiMessaging::getInstance();
    const std::string group = "token";

    assert(interface->addBlossom(group, "create", new CreateToken()));
    interface->addEndpoint("v1/token",
                           Kitsunemimi::Hanami::POST_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "create");

    assert(interface->addBlossom(group, "renew", new RenewToken()));
    interface->addEndpoint("v1/token",
                           Kitsunemimi::Hanami::PUT_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "renew");

    assert(interface->addBlossom(group, "create_internal", new CreateInternalToken()));
    interface->addEndpoint("v1/token/internal",
                           Kitsunemimi::Hanami::POST_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "create_internal");

    assert(interface->addBlossom(group, "validate", new ValidateAccess()));
    interface->addEndpoint("v1/auth",
                           Kitsunemimi::Hanami::GET_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "validate");
}

/**
 * @brief documentation endpoints
 */
void
documentationBlossomes()
{
    HanamiMessaging* interface = HanamiMessaging::getInstance();
    const std::string group = "documentation";

    assert(interface->addBlossom(group, "generate_rest_api", new GenerateRestApiDocu()));
    interface->addEndpoint("v1/documentation/api/rest",
                           Kitsunemimi::Hanami::GET_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "generate_rest_api");
}

/**
 * @brief init user endpoints
 */
void
userBlossomes()
{
    HanamiMessaging* interface = HanamiMessaging::getInstance();
    const std::string group = "user";

    assert(interface->addBlossom(group, "create", new CreateUser()));
    interface->addEndpoint("v1/user",
                           Kitsunemimi::Hanami::POST_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "create");

    assert(interface->addBlossom(group, "get", new GetUser()));
    interface->addEndpoint("v1/user",
                           Kitsunemimi::Hanami::GET_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "get");

    assert(interface->addBlossom(group, "list", new ListUsers()));
    interface->addEndpoint("v1/user/all",
                           Kitsunemimi::Hanami::GET_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "list");

    assert(interface->addBlossom(group, "delete", new DeleteUser()));
    interface->addEndpoint("v1/user",
                           Kitsunemimi::Hanami::DELETE_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "delete");

    assert(interface->addBlossom(group, "add_project", new AddProjectToUser()));
    interface->addEndpoint("v1/user/project",
                           Kitsunemimi::Hanami::POST_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "add_project");

    assert(interface->addBlossom(group, "remove_project", new RemoveProjectFromUser()));
    interface->addEndpoint("v1/user/project",
                           Kitsunemimi::Hanami::DELETE_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "remove_project");

    // TODO: move ListUserProjects-class in user-directory
    assert(interface->addBlossom(group, "list_user_projects", new ListUserProjects()));
    interface->addEndpoint("v1/user/project",
                           Kitsunemimi::Hanami::GET_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "list_user_projects");
}

/**
 * @brief init special endpoints
 */
void
projectBlossomes()
{
    HanamiMessaging* interface = HanamiMessaging::getInstance();
    const std::string group = "project";

    assert(interface->addBlossom(group, "create", new CreateProject()));
    interface->addEndpoint("v1/project",
                           Kitsunemimi::Hanami::POST_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "create");

    assert(interface->addBlossom(group, "get", new GetProject()));
    interface->addEndpoint("v1/project",
                           Kitsunemimi::Hanami::GET_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "get");

    assert(interface->addBlossom(group, "list", new ListProjects()));
    interface->addEndpoint("v1/project/all",
                           Kitsunemimi::Hanami::GET_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "list");

    assert(interface->addBlossom(group, "delete", new DeleteProject()));
    interface->addEndpoint("v1/project",
                           Kitsunemimi::Hanami::DELETE_TYPE,
                           Kitsunemimi::Hanami::BLOSSOM_TYPE,
                           group,
                           "delete");
}

void
initBlossoms()
{
    projectBlossomes();
    userBlossomes();
    documentationBlossomes();
    tokenBlossomes();
}

#endif // MISAKIGUARD_BLOSSOM_INITIALIZING_H
