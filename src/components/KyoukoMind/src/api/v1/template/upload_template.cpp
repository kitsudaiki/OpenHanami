/**
 * @file        upload_template.cpp
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

#include "upload_template.h"
#include <kyouko_root.h>

#include <libShioriArchive/datasets.h>

#include <libKitsunemimiHanamiCommon/uuid.h>
#include <libKitsunemimiHanamiCommon/enums.h>
#include <libKitsunemimiHanamiSegmentParser/segment_meta.h>

#include <libKitsunemimiCrypto/common.h>
#include <libKitsunemimiCommon/buffer/data_buffer.h>

using namespace Kitsunemimi::Hanami;

UploadTemplate::UploadTemplate()
    : Blossom("Upload a new template and store it within the database.")
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("name",
                       SAKURA_STRING_TYPE,
                       true,
                       "Name for the new template.");
    // column in database is limited to 256 characters size
    assert(addFieldBorder("name", 4, 256));
    assert(addFieldRegex("name", "[a-zA-Z][a-zA-Z_0-9]*"));

    registerInputField("template",
                       SAKURA_STRING_TYPE,
                       true,
                       "New template to upload as base64 string.");

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("uuid",
                        SAKURA_STRING_TYPE,
                        "UUID of the new uploaded template.");
    registerOutputField("name",
                        SAKURA_STRING_TYPE,
                        "Name of the new uploaded template.");
    registerOutputField("owner_id",
                        SAKURA_STRING_TYPE,
                        "ID of the user, who created the new template.");
    registerOutputField("project_id",
                        SAKURA_STRING_TYPE,
                        "ID of the project, where the new template belongs to.");
    registerOutputField("visibility",
                        SAKURA_STRING_TYPE,
                        "Visibility of the new created template (private, shared, public).");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
UploadTemplate::runTask(BlossomIO &blossomIO,
                        const Kitsunemimi::DataMap &context,
                        BlossomStatus &status,
                        Kitsunemimi::ErrorContainer &error)
{
    const std::string name = blossomIO.input.get("name").getString();
    const std::string stringContent = blossomIO.input.get("template").toString();
    const Kitsunemimi::Hanami::UserContext userContext(context);

    // check if template with the name already exist within the table
    Kitsunemimi::JsonItem getResult;
    if(KyoukoRoot::templateTable->getTemplateByName(getResult, name, userContext, error))
    {
        status.errorMessage = "Template with name '" + name + "' already exist.";
        status.statusCode = Kitsunemimi::Hanami::CONFLICT_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }
    error._errorMessages.clear();
    error._possibleSolution.clear();

    // decode base64 formated template to check if valid base64-string
    Kitsunemimi::DataBuffer convertedTemplate;
    if(Kitsunemimi::decodeBase64(convertedTemplate, stringContent) == false)
    {
        status.errorMessage = "Uploaded template is not a valid base64-String.";
        status.statusCode = Kitsunemimi::Hanami::BAD_REQUEST_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }

    // parse segment-template to validate syntax
    Kitsunemimi::Hanami::SegmentMeta parsedSegment;
    const std::string convertedTemplateStr(static_cast<const char*>(convertedTemplate.data),
                                           convertedTemplate.usedBufferSize);
    if(Kitsunemimi::Hanami::parseSegment(&parsedSegment, convertedTemplateStr, error) == false)
    {
        status.errorMessage = "Uploaded template is not a valid segment-template: \n";
        status.errorMessage += error.toString();
        status.statusCode = Kitsunemimi::Hanami::BAD_REQUEST_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }

    // convert values
    Kitsunemimi::JsonItem templateData;
    templateData.insert("name", name);
    templateData.insert("data", stringContent);
    templateData.insert("visibility", "private");

    // add new user to table
    if(KyoukoRoot::templateTable->addTemplate(templateData, userContext, error) == false)
    {
        error.addMeesage("Failed to add new template to database");
        status.statusCode = Kitsunemimi::Hanami::INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // get new created user from database
    if(KyoukoRoot::templateTable->getTemplateByName(blossomIO.output,
                                                    name,
                                                    userContext,
                                                    error) == false)
    {
        error.addMeesage("Failed to get new template from database");
        status.statusCode = Kitsunemimi::Hanami::INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    return true;
}
