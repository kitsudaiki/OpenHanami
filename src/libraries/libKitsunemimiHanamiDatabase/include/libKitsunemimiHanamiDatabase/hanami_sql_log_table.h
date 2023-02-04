/**
 * @file       hanami_sql_log_table.h
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

#ifndef KITSUNEMIMI_HANAMI_DATABASE_SQL_LOG_TABLE_H
#define KITSUNEMIMI_HANAMI_DATABASE_SQL_LOG_TABLE_H

#include <vector>
#include <string>
#include <uuid/uuid.h>

#include <libKitsunemimiCommon/logger.h>

#include <libKitsunemimiSakuraDatabase/sql_table.h>

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

class HanamiSqlLogTable
        : public Kitsunemimi::Sakura::SqlTable
{
public:
    HanamiSqlLogTable(Kitsunemimi::Sakura::SqlDatabase* db);
    virtual ~HanamiSqlLogTable();

    long getNumberOfPages(ErrorContainer &error);
    bool getPageFromDb(TableItem &resultTable,
                       const std::string &userId,
                       const uint64_t page,
                       ErrorContainer &error);
};

} // namespace Hanami
} // namespace Kitsunemimi

#endif // KITSUNEMIMI_HANAMI_DATABASE_SQL_LOG_TABLE_H
