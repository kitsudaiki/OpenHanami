/**
 * @file        template.cpp
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

#include <libHanamiAiSdk/template.h>
#include <common/http_client.h>
#include <libKitsunemimiCrypto/common.h>

namespace HanamiAI
{

/**
 * @brief upload a template to the kyouko
 *
 * @param result reference for response-message
 * @param templateName name of the new template
 * @param type type of the new template (cluster or segment)
 * @param segmentTemplate template to upload.
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
uploadTemplate(std::string &result,
               const std::string &templateName,
               const std::string &segmentTemplate,
               Kitsunemimi::ErrorContainer &error)
{
    HanamiRequest* request = HanamiRequest::getInstance();

    // convert template into base64-string
    std::string segmentTemplateB64;
    Kitsunemimi::encodeBase64(segmentTemplateB64,
                                      segmentTemplate.c_str(),
                                      segmentTemplate.size());

    // create request
    const std::string path = "/control/kyouko/v1/template/upload";
    const std::string vars = "";
    const std::string jsonBody = "{\"name\":\""
                                 + templateName
                                 + "\",\"template\":\""
                                 + segmentTemplateB64
                                 + "\"}";

    // send request
    if(request->sendPostRequest(result, path, vars, jsonBody, error) == false)
    {
        error.addMeesage("Failed to upload new template");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief get a specific template from kyouko
 *
 * @param result reference for response-message
 * @param templateUuid uuid of the template to get
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
getTemplate(std::string &result,
            const std::string &templateUuid,
            Kitsunemimi::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/kyouko/v1/template";
    const std::string vars = "uuid=" + templateUuid;

    // send request
    if(request->sendGetRequest(result, path, vars, error) == false)
    {
        error.addMeesage("Failed to get template with UUID '" + templateUuid + "'");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief list all visible templates on kyouko
 *
 * @param result reference for response-message
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
listTemplate(std::string &result,
             Kitsunemimi::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/kyouko/v1/template/all";

    // send request
    if(request->sendGetRequest(result, path, "", error) == false)
    {
        error.addMeesage("Failed to list templates");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief delete a template form kyouko
 *
 * @param result reference for response-message
 * @param templateUuid uuid of the template to delete
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
deleteTemplate(std::string &result,
               const std::string &templateUuid,
               Kitsunemimi::ErrorContainer &error)
{
    // create request
    HanamiRequest* request = HanamiRequest::getInstance();
    const std::string path = "/control/kyouko/v1/template";
    const std::string vars = "uuid=" + templateUuid;

    // send request
    if(request->sendDeleteRequest(result, path, vars, error) == false)
    {
        error.addMeesage("Failed to delete template with UUID '" + templateUuid + "'");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

} // namespace HanamiAI
