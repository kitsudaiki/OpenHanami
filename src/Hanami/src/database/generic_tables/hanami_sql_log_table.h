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

#ifndef HANAMI_DATABASE_SQL_LOG_TABLE_H
#define HANAMI_DATABASE_SQL_LOG_TABLE_H

#include <hanami_common/logger.h>
#include <hanami_database/sql_table.h>
#include <uuid/uuid.h>

#include <string>
#include <vector>

class SqlDatabase;

class HanamiSqlLogTable : public Hanami::SqlTable
{
   public:
    HanamiSqlLogTable(Hanami::SqlDatabase *db);
    virtual ~HanamiSqlLogTable();

    long getNumberOfPages(Hanami::ErrorContainer &error);
    bool getPageFromDb(Hanami::TableItem &resultTable,
                       const std::string &userId,
                       const uint64_t page,
                       Hanami::ErrorContainer &error);
};

#endif  // HANAMI_DATABASE_SQL_LOG_TABLE_H
