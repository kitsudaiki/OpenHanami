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

#include <api/endpoint_processing/blossom.h>
#include <api/endpoint_processing/http_processing/file_send.h>
#include <api/endpoint_processing/http_processing/response_builds.h>
#include <api/endpoint_processing/http_processing/string_functions.h>
#include <api/endpoint_processing/http_server.h>
#include <database/audit_log_table.h>
#include <hanami_common/logger.h>
#include <hanami_root.h>
#include <jwt-cpp/jwt.h>
// #include <jwt-cpp/traits/nlohmann-json/defaults.h>

/**
 * @brief process request and build response
 */
bool
HttpProcessing::processRequest(http::request<http::string_body>& httpRequest,
                               http::response<http::dynamic_body>& httpResponse,
                               Hanami::ErrorContainer& error)
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
    if (checkPath(path) == false) {
        error.addMessage("Path '" + path + "' is not valid");
        httpResponse.result(http::status::bad_request);
        return false;
    }

    // check if http-type is supported
    if (messageType != http::verb::get && messageType != http::verb::post
        && messageType != http::verb::put && messageType != http::verb::delete_)
    {
        httpResponse.result(http::status::bad_request);
        error.addMessage("Invalid request-method '" + std::string(httpRequest.method_string())
                         + "'");
        beast::ostream(httpResponse.body()) << error.toString();
        return false;
    }

    // check for dashboard-client-request
    if (messageType == http::verb::get && path.compare(0, 8, "/control") != 0) {
        if (processClientRequest(httpResponse, path, error) == false) {
            error.addMessage("Failed to send dashboard-files");
            return false;
        }
        return true;
    }

    // get payload of message
    if (messageType == http::verb::post || messageType == http::verb::put) {
        payload = httpRequest.body().data();
    }

    // get token from request-header
    std::string token = "";
    if (httpRequest.count("X-Auth-Token") > 0) {
        token = httpRequest.at("X-Auth-Token").to_string();
    }

    // handle control-messages
    if (cutPath(path, "/control/")) {
        HttpRequestType hType = static_cast<HttpRequestType>(messageType);
        if (processControlRequest(httpResponse, path, token, payload, hType, error) == false) {
            error.addMessage("Failed to process control-request");
            return false;
        }
        return true;
    }

    // handle default, if nothing was found
    error.addMessage("no matching endpoint found for path '" + path + "'");
    genericError_ResponseBuild(httpResponse, HttpResponseTypes::NOT_FOUND_RTYPE, error.toString());

    return false;
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
HttpProcessing::processControlRequest(http::response<http::dynamic_body>& httpResponse,
                                      const std::string& uri,
                                      const std::string& token,
                                      const std::string& inputValues,
                                      const Hanami::HttpRequestType httpType,
                                      Hanami::ErrorContainer& error)
{
    RequestMessage hanamiRequest;
    BlossomStatus status;
    json result = json::object();
    json inputValuesJson;
    std::string userId;

    do {
        // parse uri
        hanamiRequest.httpType = httpType;
        hanamiRequest.inputValues = inputValues;
        if (parseUri(token, hanamiRequest, uri, status) == false) {
            break;
        }

        // parse input-values
        try {
            inputValuesJson = json::parse(hanamiRequest.inputValues);
        }
        catch (const json::parse_error& ex) {
            status.statusCode = BAD_REQUEST_RTYPE;
            status.errorMessage = "Failed to pase input-values: " + std::string(ex.what());
            LOG_DEBUG(status.errorMessage);
            break;
        }

        // handle token-request
        if (uri == "v1/token" && hanamiRequest.httpType == Hanami::POST_TYPE) {
            inputValuesJson.erase("token");

            if (triggerBlossom(
                    result, "create", "Token", json::object(), inputValuesJson, status, error)
                == false)
            {
                error.addMessage("Token request failed");
                break;
            }
            break;
        }

        // check authentication
        json tokenData = json::object();
        json tokenInputValues = json::object();
        tokenInputValues["token"] = token;
        tokenInputValues["http_type"] = static_cast<uint32_t>(hanamiRequest.httpType);
        tokenInputValues["endpoint"] = hanamiRequest.id;
        if (triggerBlossom(
                tokenData, "validate", "Token", json::object(), tokenInputValues, status, error)
            == false)
        {
            error.addMessage("Permission-check failed");
            break;
        }

        userId = tokenData["id"];

        // convert http-type to string
        std::string httpTypeStr = "GET";
        if (hanamiRequest.httpType == Hanami::DELETE_TYPE) {
            httpTypeStr = "DELETE";
        }
        if (hanamiRequest.httpType == Hanami::GET_TYPE) {
            httpTypeStr = "GET";
        }
        if (hanamiRequest.httpType == Hanami::HEAD_TYPE) {
            httpTypeStr = "HEAD";
        }
        if (hanamiRequest.httpType == Hanami::POST_TYPE) {
            httpTypeStr = "POST";
        }
        if (hanamiRequest.httpType == Hanami::PUT_TYPE) {
            httpTypeStr = "PUT";
        }

        // write new audit-entry to database
        if (hanamiRequest.httpType != Hanami::GET_TYPE) {
            if (AuditLogTable::getInstance()->addAuditLogEntry(
                    getDatetime(), userId, hanamiRequest.id, httpTypeStr, error)
                == false)
            {
                error.addMessage("ERROR: Failed to write audit-log into database");
                status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
                break;
            }
        }

        if (hanamiRequest.id != "v1/auth") {
            inputValuesJson.erase("token");
        }

        // map endpoint to blossom
        EndpointEntry endpoint;
        if (mapEndpoint(endpoint, hanamiRequest.id, hanamiRequest.httpType) == false) {
            status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
            error.addMessage("Failed to map endpoint with id '" + hanamiRequest.id + "'");
            break;
        }

        // make real request
        if (triggerBlossom(
                result, endpoint.name, endpoint.group, tokenData, inputValuesJson, status, error)
            == false)
        {
            error.addMessage("Blossom-trigger failed");
            break;
        }

        break;
    }
    while (true);

    // build responses, based on the status-code
    if (status.statusCode != OK_RTYPE) {
        if (status.statusCode == INTERNAL_SERVER_ERROR_RTYPE) {
            return internalError_ResponseBuild(httpResponse, userId, inputValuesJson, error);
        }
        else {
            const HttpResponseTypes type = static_cast<HttpResponseTypes>(status.statusCode);
            return genericError_ResponseBuild(httpResponse, type, status.errorMessage);
        }
    }

    return success_ResponseBuild(httpResponse, result.dump());
}

/**
 * @brief check if a specific blossom was registered
 *
 * @param groupName group-identifier of the blossom
 * @param itemName item-identifier of the blossom
 *
 * @return true, if blossom with the given group- and item-name exist, else false
 */
bool
HttpProcessing::doesBlossomExist(const std::string& groupName, const std::string& itemName)
{
    auto groupIt = m_registeredBlossoms.find(groupName);
    if (groupIt != m_registeredBlossoms.end()) {
        if (groupIt->second.find(itemName) != groupIt->second.end()) {
            return true;
        }
    }

    return false;
}

/**
 * @brief SakuraLangInterface::addBlossom
 *
 * @param groupName group-identifier of the blossom
 * @param itemName item-identifier of the blossom
 * @param newBlossom pointer to the new blossom
 *
 * @return true, if blossom was registered or false, if the group- and item-name are already
 *         registered
 */
bool
HttpProcessing::addBlossom(const std::string& groupName,
                           const std::string& itemName,
                           Blossom* newBlossom)
{
    // check if already used
    if (doesBlossomExist(groupName, itemName) == true) {
        return false;
    }

    // create internal group-map, if not already exist
    auto groupIt = m_registeredBlossoms.find(groupName);
    if (groupIt == m_registeredBlossoms.end()) {
        std::map<std::string, Blossom*> newMap;
        m_registeredBlossoms.try_emplace(groupName, newMap);
    }

    // add item to group
    groupIt = m_registeredBlossoms.find(groupName);
    groupIt->second.try_emplace(itemName, newBlossom);

    return true;
}

/**
 * @brief request a registered blossom
 *
 * @param groupName group-identifier of the blossom
 * @param itemName item-identifier of the blossom
 *
 * @return pointer to the blossom or
 *         nullptr, if blossom the given group- and item-name was not found
 */
Blossom*
HttpProcessing::getBlossom(const std::string& groupName, const std::string& itemName)
{
    // search for group
    auto groupIt = m_registeredBlossoms.find(groupName);
    if (groupIt != m_registeredBlossoms.end()) {
        // search for item within group
        auto itemIt = groupIt->second.find(itemName);
        if (itemIt != groupIt->second.end()) {
            return itemIt->second;
        }
    }

    return nullptr;
}

/**
 * @brief trigger existing blossom
 *
 * @param result map with resulting items
 * @param blossomName id of the blossom to trigger
 * @param blossomGroupName id of the group of the blossom to trigger
 * @param initialValues input-values for the tree
 * @param status reference for status-output
 * @param error reference for error-output
 *
 * @return true, if successfule, else false
 */
bool
HttpProcessing::triggerBlossom(json& result,
                               const std::string& blossomName,
                               const std::string& blossomGroupName,
                               const json& context,
                               const json& initialValues,
                               BlossomStatus& status,
                               Hanami::ErrorContainer& error)
{
    LOG_DEBUG("trigger blossom");

    // get initial blossom-item
    Blossom* blossom = getBlossom(blossomGroupName, blossomName);
    if (blossom == nullptr) {
        error.addMessage("No blosom found for the id " + blossomName);
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        status.errorMessage = "";
        return false;
    }

    // inialize a new blossom-leaf for processing
    BlossomIO blossomIO;
    blossomIO.blossomName = blossomName;
    blossomIO.blossomGroupType = blossomGroupName;
    blossomIO.input = initialValues;

    // check input to be complete
    std::string errorMessage;
    if (blossom->validateFieldsCompleteness(
            initialValues, *blossom->getInputValidationMap(), FieldDef::INPUT_TYPE, errorMessage)
        == false)
    {
        status.statusCode = BAD_REQUEST_RTYPE;
        status.errorMessage = errorMessage;
        LOG_DEBUG(status.errorMessage);
        LOG_DEBUG(
            "check of completeness of input-fields failed"
            "Check of blossom '"
            + blossomName + " in group '" + blossomGroupName + "' failed.");
        return false;
    }

    // process blossom
    if (blossom->growBlossom(blossomIO, context, status, error) == false) {
        return false;
    }

    // check output to be complete
    if (blossom->validateFieldsCompleteness(blossomIO.output,
                                            *blossom->getOutputValidationMap(),
                                            FieldDef::OUTPUT_TYPE,
                                            errorMessage)
        == false)
    {
        error.addMessage(errorMessage);
        error.addMessage("check of completeness of output-fields failed");
        error.addMessage("Check of blossom '" + blossomName + " in group '" + blossomGroupName
                         + "' failed.");
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        status.errorMessage = "";
        return false;
    }

    // TODO: override only with the output-values to avoid unnecessary conflicts
    result.clear();
    overrideItems(result, blossomIO.output, ALL);

    return checkStatusCode(blossom, blossomName, blossomGroupName, status, error);
}

/**
 * @brief check if the given status-code is allowed for the endpoint
 *
 * @param blossom pointer to related blossom
 * @param blossomName name of blossom for error-message
 * @param blossomGroupName group of the blossom for error-message
 * @param status status to check
 * @param error reference for error-output
 */
bool
HttpProcessing::checkStatusCode(Blossom* blossom,
                                const std::string& blossomName,
                                const std::string& blossomGroupName,
                                BlossomStatus& status,
                                Hanami::ErrorContainer& error)
{
    if (status.statusCode == OK_RTYPE) {
        return true;
    }

    bool found = false;
    for (const uint32_t allowed : blossom->errorCodes) {
        if (allowed == status.statusCode) {
            found = true;
        }
    }

    // if given status-code is unexprected, then override it and clear the message
    // to avoid leaking unwanted information
    if (found == false) {
        error.addMessage("Status-code '" + std::to_string(status.statusCode)
                         + "' is not allowed as output for blossom '" + blossomName + "' in group '"
                         + blossomGroupName + "'");
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        status.errorMessage = "";
        return false;
    }

    return true;
}

/**
 * @brief map the endpoint to the real target
 *
 * @param result reference to the result to identify the target
 * @param id request-id
 * @param type requested http-request-type
 *
 * @return false, if mapping failes, else true
 */
bool
HttpProcessing::mapEndpoint(EndpointEntry& result,
                            const std::string& id,
                            const HttpRequestType type)
{
    const auto id_it = endpointRules.find(id);
    if (id_it != endpointRules.end()) {
        auto type_it = id_it->second.find(type);
        if (type_it != id_it->second.end()) {
            result.type = type_it->second.type;
            result.group = type_it->second.group;
            result.name = type_it->second.name;

            return true;
        }
    }

    return false;
}

/**
 * @brief add new custom-endpoint without the parser
 *
 * @param id identifier for the new entry
 * @param httpType http-type (get, post, put, delete)
 * @param sakuraType sakura-type (tree or blossom)
 * @param group blossom-group
 * @param name tree- or blossom-id
 *
 * @return false, if id together with http-type is already registered, else true
 */
bool
HttpProcessing::addEndpoint(const std::string& id,
                            const HttpRequestType& httpType,
                            const SakuraObjectType& sakuraType,
                            const std::string& group,
                            const std::string& name)
{
    EndpointEntry newEntry;
    newEntry.type = sakuraType;
    newEntry.group = group;
    newEntry.name = name;

    // search for id
    auto id_it = endpointRules.find(id);
    if (id_it != endpointRules.end()) {
        // search for http-type
        if (id_it->second.find(httpType) != id_it->second.end()) {
            return false;
        }

        // add new
        id_it->second.emplace(httpType, newEntry);
    }
    else {
        // add new
        std::map<HttpRequestType, EndpointEntry> typeEntry;
        typeEntry.emplace(httpType, newEntry);
        endpointRules.emplace(id, typeEntry);
    }

    return true;
}

/**
 * @brief override data of a data-map with new incoming information
 *
 * @param original data-map with the original key-values, which should be updates with the
 *                 information of the override-map
 * @param override map with the new incoming information
 * @param type type of override
 */
void
HttpProcessing::overrideItems(json& original, const json& override, OverrideType type)
{
    if (type == ONLY_EXISTING) {
        for (const auto& [name, item] : override.items()) {
            if (original.contains(name)) {
                original[name] = item;
            }
        }
    }
    if (type == ONLY_NON_EXISTING) {
        for (const auto& [name, item] : override.items()) {
            if (original.contains(name) == false) {
                original[name] = item;
            }
        }
    }
    else if (type == ALL) {
        for (const auto& [name, item] : override.items()) {
            original[name] = item;
        }
    }
}
