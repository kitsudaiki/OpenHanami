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

#ifndef HANAMI_SQL_TABLE_H
#define HANAMI_SQL_TABLE_H

#include <hanami_common/logger.h>
#include <hanami_common/structs.h>
#include <hanami_database/sql_table.h>
#include <uuid/uuid.h>

#include <string>
#include <vector>

class SqlDatabase;

class HanamiSqlTable : public Hanami::SqlTable
{
   public:
    HanamiSqlTable(Hanami::SqlDatabase* db);
    virtual ~HanamiSqlTable();

   protected:
    bool add(json& values, const Hanami::UserContext& userContext, Hanami::ErrorContainer& error);
    bool get(json& result,
             const Hanami::UserContext& userContext,
             std::vector<RequestCondition>& conditions,
             Hanami::ErrorContainer& error,
             const bool showHiddenValues = false);
    bool update(json& values,
                const Hanami::UserContext& userContext,
                std::vector<RequestCondition>& conditions,
                Hanami::ErrorContainer& error);
    bool getAll(Hanami::TableItem& result,
                const Hanami::UserContext& userContext,
                std::vector<RequestCondition>& conditions,
                Hanami::ErrorContainer& error,
                const bool showHiddenValues = false);
    bool del(std::vector<RequestCondition>& conditions,
             const Hanami::UserContext& userContext,
             Hanami::ErrorContainer& error);

   private:
    void fillCondition(std::vector<RequestCondition>& conditions,
                       const Hanami::UserContext& userContext);
};

#endif  // HANAMI_SQL_TABLE_H
