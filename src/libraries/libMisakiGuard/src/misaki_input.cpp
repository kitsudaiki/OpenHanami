/**
 * @file        misaki_input.cpp
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

#include <libMisakiGuard/misaki_input.h>
#include <generate_api_docu.h>

#include <libKitsunemimiJson/json_item.h>

#include <libKitsunemimiHanamiNetwork/hanami_messaging_client.h>
#include <libKitsunemimiHanamiNetwork/hanami_messaging.h>

using Kitsunemimi::Hanami::HanamiMessagingClient;
using Kitsunemimi::Hanami::HanamiMessaging;

namespace Misaki
{

/**
 * @brief init misaki-specific blossoms
 *
 * @return true, if successful, else false
 */
bool
initMisakiBlossoms()
{
    // init predefined blossoms
    HanamiMessaging* interface = HanamiMessaging::getInstance();
    const std::string group = "-";
    if(interface->addBlossom(group, "get_api_documentation", new GenerateApiDocu()) == false) {
        return false;
    }

    // add new endpoints
    if(interface->addEndpoint("v1/documentation/api",
                              Kitsunemimi::Hanami::GET_TYPE,
                              Kitsunemimi::Hanami::BLOSSOM_TYPE,
                              group,
                              "get_api_documentation") == false)
    {
        return false;
    }

    return true;
}


/**
 * @brief HanamiMessaging::getInternalToken
 *
 * @param token reference for the resulting token
 * @param componentName name of the component where the token is for
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
getInternalToken(std::string &token,
                 const std::string &componentName,
                 Kitsunemimi::ErrorContainer &error)
{
    HanamiMessagingClient* misakiClient = HanamiMessaging::getInstance()->misakiClient;
    Kitsunemimi::Hanami::ResponseMessage response;

    // create request
    Kitsunemimi::Hanami::RequestMessage request;
    request.id = "v1/token/internal";
    request.httpType = Kitsunemimi::Hanami::POST_TYPE;
    request.inputValues = "{\"service_name\":\"" + componentName + "\"}";

    // request internal jwt-token from misaki
    if(misakiClient->triggerSakuraFile(response, request, error) == false)
    {
        error.addMeesage("Failed to trigger misaki to get a internal jwt-token");
        LOG_ERROR(error);
        return false;
    }

    // check response
    if(response.success == false)
    {
        error.addMeesage("Failed to trigger misaki to get a internal jwt-token (no success)");
        LOG_ERROR(error);
        return false;
    }

    // parse response
    Kitsunemimi::JsonItem jsonItem;
    if(jsonItem.parse(response.responseContent, error) == false)
    {
        error.addMeesage("Failed to parse internal jwt-token from response of misaki");
        LOG_ERROR(error);
        return false;
    }

    // get token from response
    token = jsonItem.getItemContent()->toMap()->getStringByKey("token");
    if(token == "")
    {
        error.addMeesage("Internal jwt-token from misaki is empty");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

}
