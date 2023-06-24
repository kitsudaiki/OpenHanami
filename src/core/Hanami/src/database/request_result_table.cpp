/**
 * @file        request_result_table.cpp
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

#include <database/request_result_table.h>

#include <libKitsunemimiCommon/items/table_item.h>
#include <libKitsunemimiCommon/methods/string_methods.h>
#include <libKitsunemimiJson/json_item.h>

#include <libKitsunemimiSakuraDatabase/sql_database.h>

/**
 * @brief constructor
 *
 * @param db pointer to database
 */
RequestResultTable::RequestResultTable(Kitsunemimi::Sakura::SqlDatabase* db)
    : HanamiSqlTable(db)
{
    m_tableName = "request_result";

    DbHeaderEntry data;
    data.name = "data";
    data.hide = true;
    m_tableHeader.push_back(data);
}

/**
 * @brief destructor
 */
RequestResultTable::~RequestResultTable() {}

/**
 * @brief add new metadata of a dataset into the database
 *
 * @param data json-item with all information of the data to add to database
 * @param userContext context-object with all user specific information
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
RequestResultTable::addRequestResult(Kitsunemimi::JsonItem &data,
                                     const UserContext &userContext,
                                     Kitsunemimi::ErrorContainer &error)
{
    if(add(data, userContext, error) == false)
    {
        error.addMeesage("Failed to add snapshot to database");
        return false;
    }

    return true;
}

/**
 * @brief get a metadata-entry for a specific dataset from the database
 *
 * @param result reference for the result-output
 * @param resultUuid uuid of the data
 * @param userContext context-object with all user specific information
 * @param error reference for error-output
 * @param showHiddenValues set to true to also show as hidden marked fields
 *
 * @return true, if successful, else false
 */
bool
RequestResultTable::getRequestResult(Kitsunemimi::JsonItem &result,
                                     const std::string &resultUuid,
                                     const UserContext &userContext,
                                     Kitsunemimi::ErrorContainer &error,
                                     const bool showHiddenValues)
{
    // get user from db
    std::vector<RequestCondition> conditions;
    conditions.emplace_back("uuid", resultUuid);

    // get dataset from db
    if(get(result, userContext, conditions, error, showHiddenValues) == false)
    {
        error.addMeesage("Failed to get request-result with UUID '"
                         + resultUuid
                         + "' from database");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief get metadata of all datasets from the database
 *
 * @param result reference for the result-output
 * @param userContext context-object with all user specific information
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
RequestResultTable::getAllRequestResult(Kitsunemimi::TableItem &result,
                                        const UserContext &userContext,
                                        Kitsunemimi::ErrorContainer &error)
{
    std::vector<RequestCondition> conditions;
    if(getAll(result, userContext, conditions, error) == false)
    {
        error.addMeesage("Failed to get all request-results from database");
        return false;
    }

    return true;
}

/**
 * @brief delete metadata of a datasett from the database
 *
 * @param datasetUuid uuid of the data
 * @param userContext context-object with all user specific information
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
RequestResultTable::deleteRequestResult(const std::string &resultUuid,
                                        const UserContext &userContext,
                                        Kitsunemimi::ErrorContainer &error)
{
    std::vector<RequestCondition> conditions;
    conditions.emplace_back("uuid", resultUuid);
    if(del(conditions, userContext, error) == false)
    {
        error.addMeesage("Failed to delete request-result with UUID '"
                         + resultUuid
                         + "' from database");
        return false;
    }

    return true;
}
