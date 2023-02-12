/**
 * @file        permission.cpp
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

#include "permission.h"

#include <libKitsunemimiHanamiCommon/component_support.h>
#include <libKitsunemimiHanamiNetwork/hanami_messaging.h>
#include <libKitsunemimiHanamiNetwork/hanami_messaging_client.h>

#include <libKitsunemimiSakuraNetwork/session.h>
#include <libKitsunemimiHanamiNetwork/blossom.h>

#include <libKitsunemimiCrypto/common.h>
#include <libKitsunemimiJson/json_item.h>
#include <libKitsunemimiJwt/jwt.h>
#include <libKitsunemimiCommon/methods/string_methods.h>
#include <libKitsunemimiCommon/items/data_items.h>

namespace Kitsunemimi::Hanami
{

/**
 * @brief check if a token is valid and parse the token
 *
 * @param context reference for the parsed information of the token
 * @param token token to check
 * @param status reference for status-output
 * @param skipPermission set to true to only parse the jwt-token without check permission
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
checkPermission(DataMap &context,
                const std::string &token,
                Hanami::BlossomStatus &status,
                const bool skipPermission,
                Kitsunemimi::ErrorContainer &error)
{
    JsonItem parsedResult;

    // only get token content without validation, if misaki is not supported
    if(skipPermission)
    {
        if(token == "") {
            return true;
        }

        if(getJwtTokenPayload(parsedResult, token, error) == false)
        {
            status.statusCode = Kitsunemimi::Hanami::BAD_REQUEST_RTYPE;
            return false;
        }
    }
    else
    {
        // precheck
        if(token == "")
        {
            status.statusCode = Kitsunemimi::Hanami::BAD_REQUEST_RTYPE;
            status.errorMessage = "Token is required but missing in the request.";
            error.addMeesage("Token is missing in request");
            return false;
        }

        if(getPermission(parsedResult, token, status, error) == false)
        {
            status.statusCode = Kitsunemimi::Hanami::UNAUTHORIZED_RTYPE;
            return false;
        }
    }

    // fill context-object
    context = *parsedResult.getItemContent()->toMap();
    context.insert("token", new DataValue(token));

    return true;
}

/**
 * @brief send request to misaki to check and parse a jwt-token
 *
 * @param parsedResult reference for the parsed result
 * @param token token to check and to parse
 * @param status reference for status-output
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
getPermission(JsonItem &parsedResult,
              const std::string &token,
              Hanami::BlossomStatus &status,
              ErrorContainer &error)
{
    Kitsunemimi::Hanami::ResponseMessage responseMsg;
    Hanami::HanamiMessaging* messaging = Hanami::HanamiMessaging::getInstance();

    // create request
    Kitsunemimi::Hanami::RequestMessage requestMsg;
    requestMsg.id = "v1/auth";
    requestMsg.httpType = HttpRequestType::GET_TYPE;
    requestMsg.inputValues = "{\"token\":\"" + token + "\"}";

    if(messaging->misakiClient == nullptr)
    {
        status.statusCode = Kitsunemimi::Hanami::INTERNAL_SERVER_ERROR_RTYPE;
        error.addMeesage("Misaki is not correctly initilized.");
        return false;
    }

    // send request to misaki
    if(messaging->misakiClient->triggerSakuraFile(responseMsg, requestMsg, error) == false)
    {
        status.statusCode = Kitsunemimi::Hanami::INTERNAL_SERVER_ERROR_RTYPE;
        error.addMeesage("Unable send validation for token-request.");
        return false;
    }

    // handle failed authentication
    if(responseMsg.type == Kitsunemimi::Hanami::UNAUTHORIZED_RTYPE
            || responseMsg.success == false)
    {
        status.statusCode = responseMsg.type;
        status.errorMessage = responseMsg.responseContent;
        error.addMeesage(responseMsg.responseContent);
        return false;
    }

    if(parsedResult.parse(responseMsg.responseContent, error) == false)
    {
        status.statusCode = Kitsunemimi::Hanami::INTERNAL_SERVER_ERROR_RTYPE;
        error.addMeesage("Unable to parse auth-reponse.");
        return false;
    }

    return true;
}

}
