/**
 * @file        string_functions.h
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

#ifndef HANAMI_STRING_FUNCTIONS_H
#define HANAMI_STRING_FUNCTIONS_H

#include <api/endpoint_processing/blossom.h>
#include <hanami_common/functions/string_functions.h>
#include <hanami_common/logger.h>

#include <regex>
#include <string>

/**
 * @brief precheck path
 *
 * @param path file-path to check
 *
 * @return false, if path is invalid, else true
 */
inline bool
checkPath(const std::string& path)
{
    if (path.empty() || path[0] != '/' || path.find("..") != std::string::npos) {
        return false;
    }

    return true;
}

/**
 * @brief parse uri
 *
 * @param target reference for teh target, which has to be parsed from the uri
 * @param token jwt-token to add the the request
 * @param request reference for the request, which should be filled by values from the uri
 * @param uri uri to parse
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
inline bool
parseUri(const std::string& token,
         Hanami::RequestMessage& request,
         const std::string& uri,
         BlossomStatus& status)
{
    // first split of uri
    json parsedInputValues;
    std::vector<std::string> parts;
    Hanami::splitStringByDelimiter(parts, uri, '?');

    // check split-result
    if (parts.size() == 0) {
        status.errorMessage = "Uri is empty.";
        status.statusCode = BAD_REQUEST_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return false;
    }
    if (parts.at(0).size() <= 5) {
        status.errorMessage = "Uri is too short.";
        status.statusCode = BAD_REQUEST_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return false;
    }
    if (parts.at(0).at(0) != '/') {
        status.errorMessage = "Uri doesn't start with '/'";
        status.statusCode = BAD_REQUEST_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return false;
    }

    // parse body
    try {
        parsedInputValues = json::parse(request.inputValues);
    }
    catch (const json::parse_error& ex) {
        status.errorMessage = "Failed to parse input-values, because of json-parser error: "
                              + std::string(ex.what());
        status.statusCode = BAD_REQUEST_RTYPE;
        LOG_DEBUG(status.errorMessage);
        return false;
    }

    // get path, but without the starting "/"
    request.targetEndpoint = std::string(&parts[0][1], parts[0].size() - 1);

    // prepare payload, if exist
    if (parts.size() > 1) {
        std::vector<std::string> kvPairs;
        Hanami::splitStringByDelimiter(kvPairs, parts[1], '&');

        for (const std::string& kvPair : kvPairs) {
            const size_t cutPos = kvPair.find('=');
            const std::string key = kvPair.substr(0, cutPos);
            const std::string val = kvPair.substr(cutPos + 1, kvPair.size() - 1);

            // convert result if number and add to resulting map
            if (regex_match(val, std::regex(INT_VALUE_REGEX))) {
                parsedInputValues[key] = std::stoi(val.c_str(), NULL);
            }
            else if (regex_match(val, std::regex(FLOAT_VALUE_REGEX))) {
                parsedInputValues[key] = std::strtof(val.c_str(), NULL);
            }
            else {
                parsedInputValues[key] = val;
            }
        }
    }

    // add token to list of normal values
    parsedInputValues["token"] = token;

    request.inputValues = parsedInputValues.dump();

    return true;
}

/**
 * @brief cut first part of a string, if match
 *
 * @param path reference for the incoming path and updated path
 * @param cut part, which have to match at the beginning of the path
 *
 * @return true, if first part match, else false
 */
inline bool
cutPath(std::string& path, const std::string& cut)
{
    if (path.compare(0, cut.size(), cut) == 0) {
        path.erase(0, cut.size());
        return true;
    }

    return false;
}

#endif  // HANAMI_STRING_FUNCTIONS_H
