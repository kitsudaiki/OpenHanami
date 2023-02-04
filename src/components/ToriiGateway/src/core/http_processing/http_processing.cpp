/**
 * @file        http_processing.cpp
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

#include "http_processing.h"

#include <torii_root.h>
#include <core/http_processing/file_send.h>
#include <core/http_processing/response_builds.h>
#include <core/http_processing/string_functions.h>
#include <core/http_server.h>

#include <libKitsunemimiJson/json_item.h>
#include <libKitsunemimiCommon/logger.h>

#include <libKitsunemimiHanamiNetwork/hanami_messaging.h>
#include <libKitsunemimiHanamiNetwork/hanami_messaging_client.h>

#include <libShioriArchive/other.h>

using Kitsunemimi::Hanami::HanamiMessaging;
using Kitsunemimi::Hanami::HanamiMessagingClient;

/**
 * @brief process request and build response
 */
bool
processRequest(http::request<http::string_body> &httpRequest,
               http::response<http::dynamic_body> &httpResponse,
               Kitsunemimi::ErrorContainer &error)
{
    // build default-header for response
    httpResponse.version(httpRequest.version());
    httpResponse.keep_alive(false);
    httpResponse.set(http::field::server, "ToriiGateway");
    httpResponse.result(http::status::ok);
    httpResponse.set(http::field::content_type, "text/plain");

    // collect and prepare relevant data
    const http::verb messageType = httpRequest.method();
    std::string path = httpRequest.target().to_string();
    std::string payload = "{}";

    // Request path must be absolute and not contain "..".
    if(checkPath(path) == false)
    {
        error.addMeesage("Path '" + path + "' is not valid");
        httpResponse.result(http::status::bad_request);
        return false;
    }

    // check if http-type is supported
    if(messageType != http::verb::get
            && messageType != http::verb::post
            && messageType != http::verb::put
            && messageType != http::verb::delete_)
    {
        httpResponse.result(http::status::bad_request);
        error.addMeesage("Invalid request-method '"
                         + std::string(httpRequest.method_string())
                         + "'");
        beast::ostream(httpResponse.body()) << error.toString();
        return false;
    }

    // check for dashboard-client-request
    if(messageType == http::verb::get
            && path.compare(0, 8, "/control") != 0)
    {
        if(processClientRequest(httpResponse, path, error) == false)
        {
            error.addMeesage("Failed to send dashboard-files");
            return false;
        }
        return true;
    }

    // get payload of message
    if(messageType == http::verb::post
            || messageType == http::verb::put)
    {
        payload = httpRequest.body().data();
    }

    // get token from request-header
    std::string token = "";
    if(httpRequest.count("X-Auth-Token") > 0) {
        token = httpRequest.at("X-Auth-Token").to_string();
    }

    // handle control-messages
    if(cutPath(path, "/control/"))
    {
        HttpRequestType hType = static_cast<HttpRequestType>(messageType);
        if(processControlRequest(httpResponse, path, token, payload, hType, error) == false)
        {
            error.addMeesage("Failed to process control-request");
            return false;
        }
        return true;
    }

    // handle default, if nothing was found
    error.addMeesage("no matching endpoint found for path '" + path + "'");
    genericError_ResponseBuild(httpResponse,
                               HttpResponseTypes::NOT_FOUND_RTYPE,
                               error.toString());

    return false;
}

/**
 * @brief request token from misaki
 *
 * @param target target (misaki)
 * @param hanamiRequest hanami-request for the token-request
 * @param hanamiResponse reference for the response
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
requestToken(http::response<http::dynamic_body> &httpResponse,
             const std::string &target,
             const Kitsunemimi::Hanami::RequestMessage &hanamiRequest,
             Kitsunemimi::Hanami::ResponseMessage &hanamiResponse,
             Kitsunemimi::ErrorContainer &error)
{
    HanamiMessaging* messaging = HanamiMessaging::getInstance();
    HanamiMessagingClient* client = messaging->getOutgoingClient(target);
    if(client == nullptr)
    {
        return genericError_ResponseBuild(httpResponse,
                                          hanamiResponse.type,
                                          "Client '" + target + "' not found");
    }

    // make token-request
    if(client->triggerSakuraFile(hanamiResponse, hanamiRequest, error) == false)
    {
        return genericError_ResponseBuild(httpResponse,
                                          hanamiResponse.type,
                                          hanamiResponse.responseContent);
    }


    // handle failed authentication
    if(hanamiResponse.type == Kitsunemimi::Hanami::UNAUTHORIZED_RTYPE
            || hanamiResponse.success == false)
    {
        return genericError_ResponseBuild(httpResponse,
                                          hanamiResponse.type,
                                          hanamiResponse.responseContent);
    }

    return success_ResponseBuild(httpResponse, hanamiResponse.responseContent);
}

/**
 * @brief send request to misaki to check permissions
 *
 * @param token token to validate
 * @param component requested compoent
 * @param hanamiRequest hanami-request to the requested endpoint
 * @param responseMsg reference for the response
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
checkPermission(const std::string &token,
                const std::string &component,
                const Kitsunemimi::Hanami::RequestMessage &hanamiRequest,
                Kitsunemimi::Hanami::ResponseMessage &responseMsg,
                Kitsunemimi::ErrorContainer &error)
{
    Kitsunemimi::Hanami::RequestMessage requestMsg;

    requestMsg.id = "v1/auth";
    requestMsg.httpType = HttpRequestType::GET_TYPE;
    requestMsg.inputValues = "";

    requestMsg.inputValues.append("{\"token\":\"");
    requestMsg.inputValues.append(token);
    requestMsg.inputValues.append("\",\"component\":\"");
    requestMsg.inputValues.append(component);
    requestMsg.inputValues.append("\",\"endpoint\":\"");
    requestMsg.inputValues.append(hanamiRequest.id);
    requestMsg.inputValues.append("\",\"http_type\":");
    requestMsg.inputValues.append(std::to_string(static_cast<uint32_t>(hanamiRequest.httpType)));
    requestMsg.inputValues.append("}");

    HanamiMessaging* messaging = HanamiMessaging::getInstance();
    if(messaging->misakiClient == nullptr)
    {
        // TODO: handle error
        return false;
    }
    return messaging->misakiClient->triggerSakuraFile(responseMsg, requestMsg, error);
}

/**
 * @brief process control request
 *
 * @param uri requested uri
 * @param token given token coming from the http-header
 * @param inputValues json-formated input-values
 * @param httpType type of the http-request
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
processControlRequest(http::response<http::dynamic_body> &httpResponse,
                      const std::string &uri,
                      const std::string &token,
                      const std::string &inputValues,
                      const HttpRequestType httpType,
                      Kitsunemimi::ErrorContainer &error)
{
    std::string target = "";
    Kitsunemimi::Hanami::RequestMessage hanamiRequest;
    Kitsunemimi::Hanami::ResponseMessage hanamiResponse;
    HanamiMessaging* messaging = HanamiMessaging::getInstance();

    // parse uri
    hanamiRequest.httpType = httpType;
    hanamiRequest.inputValues = inputValues;
    if(parseUri(target, token, hanamiRequest, uri, error) == false) {
        return invalid_ResponseBuild(httpResponse, error);
    }

    // handle token-request
    if(hanamiRequest.id == "v1/token"
            && hanamiRequest.httpType == Kitsunemimi::Hanami::POST_TYPE)
    {
        return requestToken(httpResponse, target, hanamiRequest, hanamiResponse, error);
    }

    // check authentication
    if(checkPermission(token, target, hanamiRequest, hanamiResponse, error) == false) {
        return internalError_ResponseBuild(httpResponse, error);
    }

    // handle failed authentication
    if(hanamiResponse.type == Kitsunemimi::Hanami::UNAUTHORIZED_RTYPE
            || hanamiResponse.success == false)
    {
        return genericError_ResponseBuild(httpResponse,
                                          hanamiResponse.type,
                                          hanamiResponse.responseContent);
    }

    // parse response to get user-uuid of the token
    Kitsunemimi::JsonItem userData;
    if(userData.parse(hanamiResponse.responseContent, error) == false) {
        return internalError_ResponseBuild(httpResponse, error);
    }

    // send audit-message to shiori
    if(Shiori::sendAuditMessage(target,
                                hanamiRequest.id,
                                userData.get("id").getString(),
                                hanamiRequest.httpType,
                                error) == false)
    {
        error.addMeesage("Failed to send audit-log to Shiori");
        return internalError_ResponseBuild(httpResponse, error);
    }

    // forward real request
    HanamiMessagingClient* client = messaging->getOutgoingClient(target);
    if(client == nullptr)
    {
        return genericError_ResponseBuild(httpResponse,
                                          hanamiResponse.type,
                                          "Client '" + target + "' not found");
    }

    // make real request
    if(client->triggerSakuraFile(hanamiResponse, hanamiRequest, error) == false) {
        return internalError_ResponseBuild(httpResponse, error);
    }

    // handle error-response
    if(hanamiResponse.success == false)
    {
        return genericError_ResponseBuild(httpResponse,
                                          hanamiResponse.type,
                                          hanamiResponse.responseContent);
    }

    // handle success
    return success_ResponseBuild(httpResponse, hanamiResponse.responseContent);
}
