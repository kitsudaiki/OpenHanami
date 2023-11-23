/**
 * @file        tempfile_table.cpp
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

#include <database/tempfile_table.h>
#include <hanami_common/items/table_item.h>
#include <hanami_common/methods/string_methods.h>
#include <hanami_database/sql_database.h>

TempfileTable* TempfileTable::instance = nullptr;

/**
 * @brief constructor
 *
 * @param db pointer to database
 */
TempfileTable::TempfileTable() : HanamiSqlTable(Hanami::SqlDatabase::getInstance())
{
    m_tableName = "tempfile";

    DbHeaderEntry relatedResouceType;
    relatedResouceType.name = "related_resource_type";
    m_tableHeader.push_back(relatedResouceType);

    DbHeaderEntry relatedResouceUuid;
    relatedResouceUuid.name = "related_resource_uuid";
    m_tableHeader.push_back(relatedResouceUuid);

    DbHeaderEntry fileSize;
    fileSize.name = "file_size";
    fileSize.type = INT_TYPE;
    m_tableHeader.push_back(fileSize);

    DbHeaderEntry location;
    location.name = "location";
    location.hide = true;
    m_tableHeader.push_back(location);
}

/**
 * @brief destructor
 */
TempfileTable::~TempfileTable() {}

/**
 * @brief add metadata of a new tempfile to the database
 *
 * @param relatedResourceType type of the related resource of the tempfile (for example: dataset)
 * @param relatedResourceUuid uuid of the related resource of the tempfile
 * @param fileSize size of the tempfile in number of bytes
 * @param location location of the tempfile on the disc
 * @param userContext context-object with all user specific information
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
TempfileTable::addTempfile(const std::string& uuid,
                           const std::string& relatedResourceType,
                           const std::string& relatedResourceUuid,
                           const std::string& name,
                           const uint64_t fileSize,
                           const std::string& location,
                           const UserContext& userContext,
                           Hanami::ErrorContainer& error)
{
    json data;
    data["uuid"] = uuid;
    data["related_resource_type"] = relatedResourceType;
    data["related_resource_uuid"] = relatedResourceUuid;
    data["name"] = name;
    data["file_size"] = fileSize;
    data["location"] = location;
    data["visibility"] = "private";

    if (add(data, userContext, error) == false) {
        error.addMeesage("Failed to add tempfile to database");
        return false;
    }

    return true;
}

/**
 * @brief get a metadata-entry for a specific tempfile from the database
 *
 * @param result reference for the result-output
 * @param tempfileUuid uuid of the data
 * @param userContext context-object with all user specific information
 * @param error reference for error-output
 * @param showHiddenValues set to true to also show as hidden marked fields
 *
 * @return true, if successful, else false
 */
bool
TempfileTable::getTempfile(json& result,
                           const std::string& tempfileUuid,
                           const UserContext& userContext,
                           Hanami::ErrorContainer& error,
                           const bool showHiddenValues)
{
    // get user from db
    std::vector<RequestCondition> conditions;
    conditions.emplace_back("uuid", tempfileUuid);

    // get dataset from db
    if (get(result, userContext, conditions, error, showHiddenValues) == false) {
        error.addMeesage("Failed to get tempfile with UUID '" + tempfileUuid + "' from database");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief get metadata of all tempfiles from the database
 *
 * @param result reference for the result-output
 * @param userContext context-object with all user specific information
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
TempfileTable::getAllTempfile(Hanami::TableItem& result,
                              const UserContext& userContext,
                              Hanami::ErrorContainer& error)
{
    std::vector<RequestCondition> conditions;
    if (getAll(result, userContext, conditions, error) == false) {
        error.addMeesage("Failed to get all tempfiles from database");
        return false;
    }

    return true;
}

/**
 * @brief delete metadata of a tempfile from the database
 *
 * @param tempfileUuid uuid of the data
 * @param userContext context-object with all user specific information
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
TempfileTable::deleteTempfile(const std::string& tempfileUuid,
                              const UserContext& userContext,
                              Hanami::ErrorContainer& error)
{
    std::vector<RequestCondition> conditions;
    conditions.emplace_back("uuid", tempfileUuid);
    if (del(conditions, userContext, error) == false) {
        error.addMeesage("Failed to delete tempfile with UUID '" + tempfileUuid
                         + "' from database");
        return false;
    }

    return true;
}

/**
 * @brief get list of all temp-files, which are related to a specific resource
 *
 * @param relatedUuids list of templist-uuids, which are related to the requested resource
 * @param resourceType type of the related resource
 * @param resourceUuid uuid of the related resource
 * @param userContext context-object with all user specific information
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
TempfileTable::getRelatedResourceUuids(std::vector<std::string>& relatedUuids,
                                       const std::string& resourceType,
                                       const std::string& resourceUuid,
                                       const UserContext& userContext,
                                       Hanami::ErrorContainer& error)
{
    std::vector<RequestCondition> conditions;
    conditions.emplace_back("related_resource_type", resourceType);
    conditions.emplace_back("related_resource_uuid", resourceUuid);

    // get tempfile from db
    Hanami::TableItem result;
    if (getAll(result, userContext, conditions, error, true) == false) {
        error.addMeesage("Failed to get related recources for UUID '" + resourceUuid
                         + "' and type '" + resourceType + "' from database");
        return false;
    }

    for (uint64_t i = 0; i < result.getNumberOfRows(); i++) {
        relatedUuids.push_back(result.getCell(0, i));
    }

    return true;
}
