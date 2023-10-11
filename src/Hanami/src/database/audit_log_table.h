/**
 * @file        audit_log_table.h
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

#ifndef HANAMI_AUDIT_LOG_TABLE_H
#define HANAMI_AUDIT_LOG_TABLE_H

#include <database/generic_tables/hanami_sql_log_table.h>
#include <hanami_common/logger.h>

class AuditLogTable : public HanamiSqlLogTable
{
   public:
    static AuditLogTable *getInstance()
    {
        if (instance == nullptr) {
            instance = new AuditLogTable();
        }
        return instance;
    }

    ~AuditLogTable();

    bool addAuditLogEntry(const std::string &timestamp,
                          const std::string &userId,
                          const std::string &endpoint,
                          const std::string &requestType,
                          Hanami::ErrorContainer &error);
    bool getAllAuditLogEntries(Hanami::TableItem &result,
                               const std::string &userId,
                               const uint64_t page,
                               Hanami::ErrorContainer &error);

   private:
    AuditLogTable();
    static AuditLogTable *instance;
};

#endif  // HANAMI_AUDIT_LOG_TABLE_H
