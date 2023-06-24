/**
 * @file        projects_table.h
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

#ifndef MISAKIGUARD_PROJECTS_TABLE_H
#define MISAKIGUARD_PROJECTS_TABLE_H

#include <libKitsunemimiCommon/logger.h>
#include <libKitsunemimiHanamiDatabase/hanami_sql_admin_table.h>

namespace Kitsunemimi {
namespace Json {
class JsonItem;
}
}
class ProjectsTable
        : public Kitsunemimi::Hanami::HanamiSqlAdminTable
{
public:
    ProjectsTable(Kitsunemimi::Sakura::SqlDatabase* db);
    ~ProjectsTable();

    bool addProject(Kitsunemimi::JsonItem &userData,
                    Kitsunemimi::ErrorContainer &error);
    bool getProject(Kitsunemimi::JsonItem &result,
                    const std::string &projectName,
                    Kitsunemimi::ErrorContainer &error,
                    const bool showHiddenValues = false);
    bool getAllProjects(Kitsunemimi::TableItem &result,
                       Kitsunemimi::ErrorContainer &error);
    bool deleteProject(const std::string &projectName,
                       Kitsunemimi::ErrorContainer &error);
};

#endif // MISAKIGUARD_PROJECTS_TABLE_H
