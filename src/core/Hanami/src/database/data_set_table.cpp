/**
 * @file        data_set_table.cpp
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

#include <database/data_set_table.h>

#include <libKitsunemimiCommon/items/table_item.h>
#include <libKitsunemimiCommon/methods/string_methods.h>
#include <libKitsunemimiJson/json_item.h>

#include <libKitsunemimiSakuraDatabase/sql_database.h>

/**
 * @brief constructor
 *
 * @param db pointer to database
 */
DataSetTable::DataSetTable(Kitsunemimi::Sakura::SqlDatabase* db)
    : HanamiSqlTable(db)
{
    m_tableName = "data_set";

    DbHeaderEntry type;
    type.name = "type";
    type.maxLength = 64;
    m_tableHeader.push_back(type);

    DbHeaderEntry location;
    location.name = "location";
    location.hide = true;
    m_tableHeader.push_back(location);  

    DbHeaderEntry tempFiles;
    tempFiles.name = "temp_files";
    tempFiles.hide = true;
    m_tableHeader.push_back(tempFiles);
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
DataSetTable::addDataSet(Kitsunemimi::JsonItem &data,
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
 * @param datasetUuid uuid of the data
 * @param userContext context-object with all user specific information
 * @param error reference for error-output
 * @param showHiddenValues set to true to also show as hidden marked fields
 *
 * @return true, if successful, else false
 */
bool
DataSetTable::getDataSet(Kitsunemimi::JsonItem &result,
                         const std::string &datasetUuid,
                         const UserContext &userContext,
                         Kitsunemimi::ErrorContainer &error,
                         const bool showHiddenValues)
{
    // get user from db
    std::vector<RequestCondition> conditions;
    conditions.emplace_back("uuid", datasetUuid);

    // get dataset from db
    if(get(result, userContext, conditions, error, showHiddenValues) == false)
    {
        error.addMeesage("Failed to get dataset with UUID '"
                         + datasetUuid
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
DataSetTable::getAllDataSet(Kitsunemimi::TableItem &result,
                            const UserContext &userContext,
                            Kitsunemimi::ErrorContainer &error)
{
    std::vector<RequestCondition> conditions;
    if(getAll(result, userContext, conditions, error) == false)
    {
        error.addMeesage("Failed to get all datasets from database");
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
DataSetTable::deleteDataSet(const std::string &datasetUuid,
                            const UserContext &userContext,
                            Kitsunemimi::ErrorContainer &error)
{
    std::vector<RequestCondition> conditions;
    conditions.emplace_back("uuid", datasetUuid);
    if(del(conditions, userContext, error) == false)
    {
        error.addMeesage("Failed to delete dataset with UUID '"
                         + datasetUuid
                         + "' from database");
        return false;
    }

    return true;
}

/**
 * @brief update dataset in database to fully uploaded
 *
 * @param uuid uuid of the dataset
 * @param fileUuid uuid of the temporary file
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
DataSetTable::setUploadFinish(const std::string &uuid,
                              const std::string &fileUuid,
                              Kitsunemimi::ErrorContainer &error)
{
    std::vector<RequestCondition> conditions;
    conditions.emplace_back("uuid", uuid);
    Kitsunemimi::JsonItem result;

    UserContext userContext;
    userContext.isAdmin = true;

    // get dataset from db
    if(get(result, userContext, conditions, error, true) == false)
    {
        error.addMeesage("Failed to get dataset with UUID '" + uuid + "' from database");
        LOG_ERROR(error);
        return false;
    }

    // check response from database
    if(result.contains("temp_files") == false)
    {
        error.addMeesage("Entry get for the dataset with UUID '" + uuid + "' is broken");
        LOG_ERROR(error);
        return false;
    }

    // update temp-files entry to 100%
    const std::string tempFilesStr = result.get("temp_files").toString();
    Kitsunemimi::JsonItem tempFiles;
    if(tempFiles.parse(tempFilesStr, error) == false)
    {
        error.addMeesage("Failed to parse temp_files entry of dataset with UUID '"
                         + uuid
                         + "' from database");
        LOG_ERROR(error);
        return false;
    }
    tempFiles.insert(fileUuid, Kitsunemimi::JsonItem(1.0f), true);

    // update new entry within the database
    Kitsunemimi::JsonItem newValues;
    newValues.insert("temp_files", Kitsunemimi::JsonItem(tempFiles.toString()));
    if(update(newValues, userContext, conditions, error) == false)
    {
        error.addMeesage("Failed to update entry of dataset with UUID '" + uuid + "' in database");
        LOG_ERROR(error);
        return false;
    }

    return true;
}
