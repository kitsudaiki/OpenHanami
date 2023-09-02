/**
 * @file        users_database.h
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

#ifndef HANAMI_USERS_TABLE_H
#define HANAMI_USERS_TABLE_H

#include <hanami_common/logger.h>
#include <database/generic_tables/hanami_sql_admin_table.h>

namespace Hanami {
class JsonItem;
}
class UsersTable
        : public HanamiSqlAdminTable
{
public:
    static UsersTable* getInstance()
    {
        if(instance == nullptr) {
            instance = new UsersTable();
        }
        return instance;
    }

    ~UsersTable();

    bool initNewAdminUser(Hanami::ErrorContainer &error);

    bool addUser(Hanami::JsonItem &userData,
                 Hanami::ErrorContainer &error);
    bool getUser(Hanami::JsonItem &result,
                 const std::string &userId,
                 Hanami::ErrorContainer &error,
                 const bool showHiddenValues);
    bool getAllUser(Hanami::TableItem &result,
                    Hanami::ErrorContainer &error);
    bool deleteUser(const std::string &userId,
                    Hanami::ErrorContainer &error);
    bool updateProjectsOfUser(const std::string &userId,
                              const std::string &newProjects,
                              Hanami::ErrorContainer &error);

private:
    UsersTable();
    static UsersTable* instance;

    bool getEnvVar(std::string &content, const std::string &key) const;

    bool getAllAdminUser(Hanami::ErrorContainer &error);
};

#endif // HANAMI_USERS_TABLE_H
