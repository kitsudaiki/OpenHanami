/**
 * @file       hanami_sql_table.cpp
 *
 * @author     Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright  Apache License Version 2.0
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

#include "hanami_sql_table.h"

#include <hanami_database/sql_database.h>

#include <hanami_common/methods/string_methods.h>
#include <hanami_json/json_item.h>

#include <uuid/uuid.h>

/**
 * @brief constructor, which add basic columns to the table
 *
 * @param db pointer to database
 */
HanamiSqlTable::HanamiSqlTable(Hanami::SqlDatabase* db)
    : SqlTable(db)
{
    DbHeaderEntry uuid;
    uuid.name = "uuid";
    uuid.maxLength = 36;
    uuid.isPrimary = true;
    m_tableHeader.push_back(uuid);

    DbHeaderEntry projectId;
    projectId.name = "project_id";
    projectId.maxLength = 256;
    m_tableHeader.push_back(projectId);

    DbHeaderEntry ownerId;
    ownerId.name = "owner_id";
    ownerId.maxLength = 256;
    m_tableHeader.push_back(ownerId);

    DbHeaderEntry visibility;
    visibility.name = "visibility";
    visibility.maxLength = 10;
    m_tableHeader.push_back(visibility);

    DbHeaderEntry name;
    name.name = "name";
    name.maxLength = 256;
    m_tableHeader.push_back(name);
}

/**
 * @brief destructor
 */
HanamiSqlTable::~HanamiSqlTable() {}

/**
 * @brief add a new row to the table
 *
 * @param values json-item with key-value-pairs, which should be added as new row to the table
 * @param userContext context-object with all user specific information
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
HanamiSqlTable::add(Hanami::JsonItem &values,
                    const UserContext &userContext,
                    Hanami::ErrorContainer &error)
{
    // generate new uuid if the is no predefined
    if(values.contains("uuid") == false)
    {
        // create uuid
        char uuid[UUID_STR_LEN];
        uuid_t binaryUuid;
        uuid_generate_random(binaryUuid);
        uuid_unparse_lower(binaryUuid, uuid);

        // fill into string, but must be reduced by 1 to remove the escate-character
        std::string uuidString = std::string(uuid, UUID_STR_LEN - 1);
        Hanami::toLowerCase(uuidString);
        values.insert("uuid", uuidString);
    }

    // add user-ids
    values.insert("owner_id", userContext.userId, true);
    values.insert("project_id", userContext.projectId, true);

    return insertToDb(values, error);
}

/**
 * @brief get specific values for the table
 *
 * @param result reference for result-output
 * @param userContext context-object with all user specific information
 * @param conditions list of conditions to filter result
 * @param error reference for error-output
 * @param showHiddenValues true to also return as hidden marked values
 *
 * @return true, if successful, else false
 */
bool
HanamiSqlTable::get(Hanami::JsonItem &result,
                    const UserContext &userContext,
                    std::vector<RequestCondition> &conditions,
                    Hanami::ErrorContainer &error,
                    const bool showHiddenValues)
{
    fillCondition(conditions, userContext);
    return getFromDb(result, conditions, error, showHiddenValues);
}

/**
 * @brief update specific values for the table
 *
 * @param values key-values-pairs to update
 * @param userContext context-object with all user specific information
 * @param conditions list of conditions to filter result
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
HanamiSqlTable::update(Hanami::JsonItem &values,
                       const UserContext &userContext,
                       std::vector<RequestCondition> &conditions,
                       Hanami::ErrorContainer &error)
{
    fillCondition(conditions, userContext);
    return updateInDb(conditions, values, error);
}

/**
 * @brief get all entries of the table
 *
 * @param result reference for result-output
 * @param userContext context-object with all user specific information
 * @param conditions predefined list of conditions for filtering
 * @param error reference for error-output
 * @param showHiddenValues true to also return as hidden marked values
 *
 * @return true, if successful, else false
 */
bool
HanamiSqlTable::getAll(Hanami::TableItem &result,
                       const UserContext &userContext,
                       std::vector<RequestCondition> &conditions,
                       Hanami::ErrorContainer &error,
                       const bool showHiddenValues)
{
    fillCondition(conditions, userContext);
    return getFromDb(result, conditions, error, showHiddenValues);
}

/**
 * @brief HanamiSqlTable::del
 *
 * @param conditions list of conditions to filter result
 * @param userContext context-object with all user specific information
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
HanamiSqlTable::del(std::vector<RequestCondition> &conditions,
                    const UserContext &userContext,
                    Hanami::ErrorContainer &error)
{
    fillCondition(conditions, userContext);
    return deleteFromDb(conditions, error);
}

/**
 * @brief update list of conditions based on admin-status
 *
 * @param conditions list of conditions to filter result
 * @param userContext context-object with all user specific information
 */
void
HanamiSqlTable::fillCondition(std::vector<RequestCondition> &conditions,
                              const UserContext &userContext)
{
    if(userContext.isAdmin) {
        return;
    }

    if(userContext.isProjectAdmin)
    {
        conditions.emplace_back("project_id", userContext.projectId);
        return;
    }

    conditions.emplace_back("owner_id", userContext.userId);
    conditions.emplace_back("project_id", userContext.projectId);

    return;
}
