/**
 * @file        generate_rest_api_docu.cpp
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

#include "generate_rest_api_docu.h"

#include <hanami_root.h>
#include <api/endpoint_processing/blossom.h>

#include <libKitsunemimiCrypto/common.h>
#include <libKitsunemimiCommon/methods/string_methods.h>
#include <libKitsunemimiCommon/methods/file_methods.h>
#include <libKitsunemimiJson/json_item.h>

std::map<HttpResponseTypes, std::string> responseMessage =
{
    {OK_RTYPE, "Successful response."},
    {CONFLICT_RTYPE, "Resource with name or id already exist."},
    {BAD_REQUEST_RTYPE, "Request has invalid syntax."},
    {UNAUTHORIZED_RTYPE, "Information for authentication are missing or are invalid"},
    {NOT_FOUND_RTYPE, "The requested resource was not found."},
    {INTERNAL_SERVER_ERROR_RTYPE, "Something internally went wrong. "
                                  "Check the server-internal logs for more information."},
};

/**
 * @brief createMdDocumentation
 * @param docu
 */
void
createOpenApiDocumentation(std::string &docu)
{
    Kitsunemimi::JsonItem result;
    result.insert("openapi", "3.0.0");

    Kitsunemimi::JsonItem info;
    info.insert("title", "API documentation");
    info.insert("version", "unreleased");
    result.insert("info", info);

    Kitsunemimi::JsonItem contact;
    info.insert("name", "Tobias Anker");
    info.insert("email", "tobias.anker@kitsunemimi.moe");
    result.insert("contact", contact);

    Kitsunemimi::JsonItem license;
    license.insert("name", "Apache 2.0");
    license.insert("url", "https://www.apache.org/licenses/LICENSE-2.0.html");
    result.insert("license", license);

    Kitsunemimi::JsonItem paths;
    generateEndpointDocu_openapi(paths);
    result.insert("paths", paths);

    docu = result.toString(true);
}

/**
 * @brief addErrorCodes
 * @param responses
 * @param allowedErrorCodes
 */
void
addResponsess(Kitsunemimi::JsonItem &responses,
              Blossom* blossom)
{
    Kitsunemimi::JsonItem resp200;
    resp200.insert("description", responseMessage[OK_RTYPE]);
    Kitsunemimi::JsonItem content;
    Kitsunemimi::JsonItem jsonApplication;
    Kitsunemimi::JsonItem schema;
    schema.insert("type", "object");
    createBodyParams_openapi(schema, blossom->getOutputValidationMap(), false);
    jsonApplication.insert("schema", schema);
    content.insert("application/json", jsonApplication);
    resp200.insert("content", content);
    responses.insert("200", resp200);

    for(const HttpResponseTypes code : blossom->errorCodes)
    {
        Kitsunemimi::JsonItem errorResponse;
        errorResponse.insert("description", responseMessage[code]);
        Kitsunemimi::JsonItem content;
        Kitsunemimi::JsonItem jsonApplication;
        Kitsunemimi::JsonItem schema;
        schema.insert("type", "string");
        jsonApplication.insert("schema", schema);
        content.insert("text/plain", jsonApplication);
        errorResponse.insert("content", content);
        responses.insert(std::to_string(code), errorResponse);
    }
}

/**
 * @brief addTokenRequirement
 * @param parameters
 */
void
addTokenRequirement(Kitsunemimi::JsonItem &parameters)
{
    Kitsunemimi::JsonItem param;
    param.insert("in", "header");
    param.insert("description", "JWT-Token for authentication");
    param.insert("name", "X-Auth-Token");
    param.insert("required", true);

    Kitsunemimi::JsonItem schema;
    schema.insert("type","string");

    param.insert("schema", schema);
    parameters.append(param);
}

/**
 * @brief createQueryParams_openapi
 * @param schema
 * @param defMap
 * @param isRequest
 */
void
createQueryParams_openapi(Kitsunemimi::JsonItem &parameters,
                          const std::map<std::string, FieldDef>* defMap)
{
    for(const auto& [field, fieldDef] : *defMap)
    {
        const FieldType fieldType = fieldDef.fieldType;
        const std::string comment = fieldDef.comment;
        const bool isRequired = fieldDef.isRequired;
        const Kitsunemimi::DataItem* defaultVal = fieldDef.defaultVal;
        const Kitsunemimi::DataItem* matchVal = fieldDef.match;
        std::string regexVal = fieldDef.regex;
        const long lowerLimit = fieldDef.lowerLimit;
        const long upperLimit = fieldDef.upperLimit;

        Kitsunemimi::JsonItem param;
        param.insert("in", "query");
        param.insert("name", field);

        // required
        if(isRequired) {
            param.insert("required", isRequired);
        }

        // comment
        if(comment != "") {
            param.insert("description", comment);
        }

        Kitsunemimi::JsonItem schema;

        // type
        if(fieldType == SAKURA_MAP_TYPE) {
            schema.insert("type","object");
        } else if(fieldType == SAKURA_ARRAY_TYPE) {
            schema.insert("type","array");
        } else if(fieldType == SAKURA_BOOL_TYPE) {
            schema.insert("type","boolean");
        } else if(fieldType == SAKURA_INT_TYPE) {
            schema.insert("type","integer");
        } else if(fieldType == SAKURA_FLOAT_TYPE) {
            schema.insert("type","number");
        } else if(fieldType == SAKURA_STRING_TYPE) {
            schema.insert("type","string");
        }

        // default
        if(defaultVal != nullptr
                && isRequired == false)
        {
            schema.insert("default", defaultVal->toString());
        }

        // match
        if(regexVal != "")
        {
            Kitsunemimi::replaceSubstring(regexVal, "\\", "\\\\");
            schema.insert("pattern", regexVal);
        }

        // border
        if(lowerLimit != 0
                || upperLimit != 0)
        {
            if(fieldType == SAKURA_INT_TYPE)
            {
                schema.insert("minimum", std::to_string(lowerLimit));
                schema.insert("maximum", std::to_string(upperLimit));
            }
            if(fieldType == SAKURA_STRING_TYPE)
            {
                schema.insert("minLength", std::to_string(lowerLimit));
                schema.insert("maxLength", std::to_string(upperLimit));
            }
        }

        // match
        if(matchVal != nullptr)
        {
            Kitsunemimi::JsonItem match;
            std::string content = matchVal->toString();
            Kitsunemimi::replaceSubstring(content, "\"", "\\\"");
            match.append(content);
            schema.insert("enum", match);
        }

        param.insert("schema", schema);
        parameters.append(param);
    }
}

/**
 * @brief generate documenation for all fields
 *
 * @param docu reference to the complete document
 * @param defMap map with all field to ducument
 * @param isRequest true to say that the actual field is a request-field
 */
void
createBodyParams_openapi(Kitsunemimi::JsonItem &schema,
                         const std::map<std::string, FieldDef>* defMap,
                         const bool isRequest)
{
    std::vector<std::string> requiredFields;

    Kitsunemimi::JsonItem properties;
    for(const auto& [id, fieldDef] : *defMap)
    {
        Kitsunemimi::JsonItem temp;

        const std::string field = id;
        const FieldType fieldType = fieldDef.fieldType;
        const std::string comment = fieldDef.comment;
        const bool isRequired = fieldDef.isRequired;
        const Kitsunemimi::DataItem* defaultVal = fieldDef.defaultVal;
        const Kitsunemimi::DataItem* matchVal = fieldDef.match;
        std::string regexVal = fieldDef.regex;
        const long lowerLimit = fieldDef.lowerLimit;
        const long upperLimit = fieldDef.upperLimit;

        // type
        if(fieldType == SAKURA_MAP_TYPE) {
            temp.insert("type","object");
        } else if(fieldType == SAKURA_ARRAY_TYPE) {
            temp.insert("type","array");
            Kitsunemimi::JsonItem array;
            array.insert("type", "string");

            // match
            if(matchVal != nullptr)
            {
                Kitsunemimi::JsonItem match;
                Kitsunemimi::ErrorContainer error;
                match.parse(matchVal->toString(), error);
                array.insert("enum", match);
            }

            temp.insert("items", array);
        } else if(fieldType == SAKURA_BOOL_TYPE) {
            temp.insert("type","boolean");
        } else if(fieldType == SAKURA_INT_TYPE) {
            temp.insert("type","integer");
        } else if(fieldType == SAKURA_FLOAT_TYPE) {
            temp.insert("type","number");
        } else if(fieldType == SAKURA_STRING_TYPE) {
            temp.insert("type","string");
        }

        // comment
        if(comment != "") {
            temp.insert("description", comment);
        }

        if(isRequest)
        {
            // required
            if(isRequired) {
                requiredFields.push_back(field);
            }

            // default
            if(defaultVal != nullptr
                    && isRequired == false)
            {
                temp.insert("default", defaultVal->toString());
            }

            // match
            if(regexVal != "")
            {
                Kitsunemimi::replaceSubstring(regexVal, "\\", "\\\\");
                temp.insert("pattern", regexVal);
            }

            // border
            if(lowerLimit != 0
                    || upperLimit != 0)
            {
                if(fieldType == SAKURA_INT_TYPE)
                {
                    temp.insert("minimum", std::to_string(lowerLimit));
                    temp.insert("maximum", std::to_string(upperLimit));
                }
                if(fieldType == SAKURA_STRING_TYPE)
                {
                    temp.insert("minLength", std::to_string(lowerLimit));
                    temp.insert("maxLength", std::to_string(upperLimit));
                }
            }

            // match
            if(matchVal != nullptr)
            {
                Kitsunemimi::JsonItem match;
                std::string content = matchVal->toString();
                Kitsunemimi::replaceSubstring(content, "\"", "\\\"");
                match.append(content);
                temp.insert("enum", match);
            }

        }

        properties.insert(field, temp);
    }

    schema.insert("properties", properties);

    if(isRequest)
    {
        Kitsunemimi::JsonItem required;
        for(const std::string& field : requiredFields) {
            required.append(field);
        }
        schema.insert("required", required);
    }
}

/**
 * @brief generate documentation for the endpoints
 *
 * @param docu reference to the complete document
 */
void
generateEndpointDocu_openapi(Kitsunemimi::JsonItem &result)
{
    for(const auto& [endpointPath, httpDef] : HanamiRoot::root->endpointRules)
    {
        // add endpoint
        Kitsunemimi::JsonItem endpoint;

        for(const auto& [type, endpointEntry] : httpDef)
        {
            Kitsunemimi::JsonItem endpointType;

            Blossom* blossom = HanamiRoot::root->getBlossom(endpointEntry.group,
                                                            endpointEntry.name);
            if(blossom == nullptr) {
                // TODO: handle error
                return;
            }

            // add comment/describtion
            endpointType.insert("summary", blossom->comment);

            Kitsunemimi::JsonItem tags;
            tags.append(endpointEntry.group);
            endpointType.insert("tags", tags);

            Kitsunemimi::JsonItem parameters;

            if(blossom->requiresAuthToken) {
                addTokenRequirement(parameters);
            }

            if(type == POST_TYPE
                    || type == PUT_TYPE)
            {
                Kitsunemimi::JsonItem requestBody;
                requestBody.insert("required", true);
                Kitsunemimi::JsonItem content;
                Kitsunemimi::JsonItem jsonApplication;
                Kitsunemimi::JsonItem schema;
                schema.insert("type", "object");
                createBodyParams_openapi(schema, blossom->getInputValidationMap(), true);
                jsonApplication.insert("schema", schema);
                content.insert("application/json", jsonApplication);
                requestBody.insert("content", content);
                endpointType.insert("requestBody", requestBody);
            }

            if(type == GET_TYPE
                    || type == DELETE_TYPE)
            {
                createQueryParams_openapi(parameters, blossom->getInputValidationMap());
            }
            endpointType.insert("parameters", parameters);

            // add response-codes
            Kitsunemimi::JsonItem responses;
            addResponsess(responses, blossom);
            endpointType.insert("responses", responses);

            // add http-type
            if(type == GET_TYPE) {
                endpoint.insert("get", endpointType);
            } else if(type == POST_TYPE) {
                endpoint.insert("post", endpointType);
            } else if(type == DELETE_TYPE) {
                endpoint.insert("delete", endpointType);
            } else if(type == PUT_TYPE) {
                endpoint.insert("put", endpointType);
            }
        }

        result.insert(endpointPath, endpoint);
    }
}
