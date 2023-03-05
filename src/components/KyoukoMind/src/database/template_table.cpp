/**
 * @file        cluster_table.cpp
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

#include "template_table.h"

#include <libKitsunemimiCommon/items/table_item.h>
#include <libKitsunemimiCommon/methods/string_methods.h>
#include <libKitsunemimiJson/json_item.h>

#include <libKitsunemimiSakuraDatabase/sql_database.h>

/**
 * @brief constructor
 */
TemplateTable::TemplateTable(Kitsunemimi::Sakura::SqlDatabase* db)
    : HanamiSqlTable(db)
{
    m_tableName = "templates";

    DbHeaderEntry templateString;
    templateString.name = "data";
    templateString.hide = true;
    m_tableHeader.push_back(templateString);
}

/**
 * @brief destructor
 */
TemplateTable::~TemplateTable() {}

/**
 * @brief add a new template to the database
 *
 * @param userData json-item with all information of the cluster to add to database
 * @param userContext context-object with all user specific information
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
TemplateTable::addTemplate(Kitsunemimi::JsonItem &clusterData,
                           const Kitsunemimi::Hanami::UserContext &userContext,
                           Kitsunemimi::ErrorContainer &error)
{
    if(add(clusterData, userContext, error) == false)
    {
        error.addMeesage("Failed to add template to database");
        return false;
    }

    return true;
}

/**
 * @brief get a template from the database by his name
 *
 * @param result reference for the result-output in case that a cluster with this name was found
 * @param templateUuid uuid of the requested template
 * @param userContext context-object with all user specific information
 * @param error reference for error-output
 * @param showHiddenValues set to true to also show as hidden marked fields
 *
 * @return true, if successful, else false
 */
bool
TemplateTable::getTemplate(Kitsunemimi::JsonItem &result,
                           const std::string &templateUuid,
                           const Kitsunemimi::Hanami::UserContext &userContext,
                           Kitsunemimi::ErrorContainer &error,
                           const bool showHiddenValues)
{
    std::vector<RequestCondition> conditions;
    conditions.emplace_back("uuid", templateUuid);

    // get user from db
    if(get(result, userContext, conditions, error, showHiddenValues) == false)
    {
        error.addMeesage("Failed to get template with UUID '"
                         + templateUuid
                         + "' from database");
        return false;
    }

    return true;
}

/**
 * @brief get a template from the database by his name
 *
 * @param result reference for the result-output in case that a cluster with this name was found
 * @param templateName name of the requested template
 * @param userContext context-object with all user specific information
 * @param error reference for error-output
 * @param showHiddenValues set to true to also show as hidden marked fields
 *
 * @return true, if successful, else false
 */
bool
TemplateTable::getTemplateByName(Kitsunemimi::JsonItem &result,
                                 const std::string &templateName,
                                 const Kitsunemimi::Hanami::UserContext &userContext,
                                 Kitsunemimi::ErrorContainer &error,
                                 const bool showHiddenValues)
{
    std::vector<RequestCondition> conditions;
    conditions.emplace_back("name", templateName);

    // get user from db
    if(get(result, userContext, conditions, error, showHiddenValues) == false)
    {
        error.addMeesage("Failed to get template from database by name '" + templateName + "'");
        return false;
    }

    return true;
}

/**
 * @brief get all templates from the database table
 *
 * @param result reference for the result-output
 * @param userContext context-object with all user specific information
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
TemplateTable::getAllTemplate(Kitsunemimi::TableItem &result,
                              const Kitsunemimi::Hanami::UserContext &userContext,
                              Kitsunemimi::ErrorContainer &error)
{
    std::vector<RequestCondition> conditions;
    if(getAll(result, userContext, conditions, error) == false)
    {
        error.addMeesage("Failed to get all templates from database");
        return false;
    }

    return true;
}

/**
 * @brief delete a cluster from the table
 *
 * @param templateUuid uuid of the template to delete
 * @param userContext context-object with all user specific information
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
TemplateTable::deleteTemplate(const std::string &templateUuid,
                              const Kitsunemimi::Hanami::UserContext &userContext,
                              Kitsunemimi::ErrorContainer &error)
{
    std::vector<RequestCondition> conditions;
    conditions.emplace_back("uuid", templateUuid);

    if(del(conditions, userContext, error) == false)
    {
        error.addMeesage("Failed to delete template with UUID '"
                         + templateUuid
                         + "' from database");
        return false;
    }

    return true;
}
