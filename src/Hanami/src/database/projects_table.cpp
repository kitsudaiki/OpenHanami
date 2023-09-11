/**
 * @file        projects_table.cpp
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

#include <database/projects_table.h>

#include <hanami_common/items/table_item.h>
#include <hanami_common/methods/string_methods.h>
#include <hanami_crypto/hashes.h>
#include <hanami_database/sql_database.h>

ProjectsTable* ProjectsTable::instance = nullptr;

/**
 * @brief constructor
 */
ProjectsTable::ProjectsTable()
    : HanamiSqlAdminTable(Hanami::SqlDatabase::getInstance())
{
    m_tableName = "projects";
}

/**
 * @brief destructor
 */
ProjectsTable::~ProjectsTable() {}

/**
 * @brief add a new project to the database
 *
 * @param userData json-item with all information of the project to add to database
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
ProjectsTable::addProject(json &userData,
                          Hanami::ErrorContainer &error)
{
    if(insertToDb(userData, error) == false)
    {
        error.addMeesage("Failed to add user to database");
        return false;
    }

    return true;
}

/**
 * @brief get a project from the database by its id
 *
 * @param result reference for the result-output in case that a project with this name was found
 * @param projectId id of the requested project
 * @param error reference for error-output
 * @param showHiddenValues set to true to also show as hidden marked fields
 *
 * @return true, if successful, else false
 */
bool
ProjectsTable::getProject(json &result,
                          const std::string &projectId,
                          Hanami::ErrorContainer &error,
                          const bool showHiddenValues)
{
    std::vector<RequestCondition> conditions;
    conditions.emplace_back("id", projectId);

    if(getFromDb(result, conditions, error, showHiddenValues) == false)
    {
        error.addMeesage("Failed to get user with id '"
                         + projectId
                         + "' from database");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief get all projects from the database table
 *
 * @param result reference for the result-output
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
ProjectsTable::getAllProjects(Hanami::TableItem &result,
                              Hanami::ErrorContainer &error)
{
    std::vector<RequestCondition> conditions;
    if(getFromDb(result, conditions, error, false) == false)
    {
        error.addMeesage("Failed to get all users from database");
        return false;
    }

    return true;
}

/**
 * @brief delete a project from the table
 *
 * @param projectId id of the project to delete
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
ProjectsTable::deleteProject(const std::string &projectId,
                             Hanami::ErrorContainer &error)
{
    std::vector<RequestCondition> conditions;
    conditions.emplace_back("id", projectId);

    if(deleteFromDb(conditions, error) == false)
    {
        error.addMeesage("Failed to delete user with id '"
                         + projectId
                         + "' from database");
        return false;
    }

    return true;
}
