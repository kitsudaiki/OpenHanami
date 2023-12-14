/**
 * @file        dataset_table.cpp
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

#include <database/dataset_table.h>
#include <hanami_common/items/table_item.h>
#include <hanami_common/methods/string_methods.h>
#include <hanami_database/sql_database.h>
#include <hanami_files/dataset_files/dataset_functions.h>

DataSetTable* DataSetTable::instance = nullptr;

/**
 * @brief constructor
 *
 * @param db pointer to database
 */
DataSetTable::DataSetTable() : HanamiSqlTable(Hanami::SqlDatabase::getInstance())
{
    m_tableName = "dataset";

    DbHeaderEntry type;
    type.name = "type";
    type.maxLength = 64;
    m_tableHeader.push_back(type);

    DbHeaderEntry location;
    location.name = "location";
    location.hide = true;
    m_tableHeader.push_back(location);
}

/**
 * @brief destructor
 */
DataSetTable::~DataSetTable() {}

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
DataSetTable::addDataSet(json& data, const UserContext& userContext, Hanami::ErrorContainer& error)
{
    if (add(data, userContext, error) == false) {
        error.addMessage("Failed to add checkpoint to database");
        return false;
    }

    return true;
}

/**
 * @brief get a metadata-entry for a specific dataset from the database
 *
 * @param result reference for the result-output
 * @param datasetUuid uuid of the data
 * @param userContext context-object with all user specific information
 * @param error reference for error-output
 * @param showHiddenValues set to true to also show as hidden marked fields
 *
 * @return true, if successful, else false
 */
bool
DataSetTable::getDataSet(json& result,
                         const std::string& datasetUuid,
                         const UserContext& userContext,
                         Hanami::ErrorContainer& error,
                         const bool showHiddenValues)
{
    // get user from db
    std::vector<RequestCondition> conditions;
    conditions.emplace_back("uuid", datasetUuid);

    // get dataset from db
    if (get(result, userContext, conditions, error, showHiddenValues) == false) {
        error.addMessage("Failed to get dataset with UUID '" + datasetUuid + "' from database");
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
DataSetTable::getAllDataSet(Hanami::TableItem& result,
                            const UserContext& userContext,
                            Hanami::ErrorContainer& error)
{
    std::vector<RequestCondition> conditions;
    if (getAll(result, userContext, conditions, error) == false) {
        error.addMessage("Failed to get all datasets from database");
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
DataSetTable::deleteDataSet(const std::string& datasetUuid,
                            const UserContext& userContext,
                            Hanami::ErrorContainer& error)
{
    std::vector<RequestCondition> conditions;
    conditions.emplace_back("uuid", datasetUuid);
    if (del(conditions, userContext, error) == false) {
        error.addMessage("Failed to delete dataset with UUID '" + datasetUuid + "' from database");
        return false;
    }

    return true;
}

/**
 * @brief getDateSetInfo
 * @param dataUuid
 * @param error
 * @return
 */
bool
DataSetTable::getDateSetInfo(json& result,
                             const std::string& dataUuid,
                             const json& context,
                             Hanami::ErrorContainer& error)
{
    const UserContext userContext(context);

    if (getDataSet(result, dataUuid, userContext, error, true) == false) {
        return false;
    }

    // get file information
    const std::string location = result["location"];
    if (getHeaderInformation(result, location, error) == false) {
        error.addMessage("Failed the read information from file '" + location + "'");
        return false;
    }

    return true;
}
