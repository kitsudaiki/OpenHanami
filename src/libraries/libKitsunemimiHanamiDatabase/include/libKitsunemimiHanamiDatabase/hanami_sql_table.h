/**
 * @file       hanami_sql_table.h
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

#ifndef KITSUNEMIMI_HANAMI_DATABASE_SQL_TABLE_H
#define KITSUNEMIMI_HANAMI_DATABASE_SQL_TABLE_H

#include <vector>
#include <string>
#include <uuid/uuid.h>

#include <libKitsunemimiCommon/logger.h>

#include <libKitsunemimiSakuraDatabase/sql_table.h>

#include <libKitsunemimiHanamiCommon/structs.h>

namespace Kitsunemimi {
namespace Json {
class JsonItem;
}
}

namespace Kitsunemimi
{
namespace Hanami
{
class SqlDatabase;

class HanamiSqlTable
        : public Kitsunemimi::Sakura::SqlTable
{
public:
    HanamiSqlTable(Kitsunemimi::Sakura::SqlDatabase* db);
    virtual ~HanamiSqlTable();

    bool add(JsonItem &values,
             const Kitsunemimi::Hanami::UserContext &userContext,
             Kitsunemimi::ErrorContainer &error);
    bool get(JsonItem &result,
             const Kitsunemimi::Hanami::UserContext &userContext,
             std::vector<RequestCondition> &conditions,
             Kitsunemimi::ErrorContainer &error,
             const bool showHiddenValues = false);
    bool update(JsonItem &values,
                const Kitsunemimi::Hanami::UserContext &userContext,
                std::vector<RequestCondition> &conditions,
                Kitsunemimi::ErrorContainer &error);
    bool getAll(Kitsunemimi::TableItem &result,
                const Kitsunemimi::Hanami::UserContext &userContext,
                std::vector<RequestCondition> &conditions,
                Kitsunemimi::ErrorContainer &error,
                const bool showHiddenValues = false);
    bool del(std::vector<RequestCondition> &conditions,
             const Kitsunemimi::Hanami::UserContext &userContext,
             Kitsunemimi::ErrorContainer &error);

private:
    void fillCondition(std::vector<RequestCondition> &conditions,
                       const Kitsunemimi::Hanami::UserContext &userContext);
};

} // namespace Hanami
} // namespace Kitsunemimi

#endif // KITSUNEMIMI_HANAMI_DATABASE_SQL_TABLE_H
