/**
 * @file        request_result_table.h
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

#ifndef HANAMI_REQUEST_RESULT_TABLE_H
#define HANAMI_REQUEST_RESULT_TABLE_H

#include <database/generic_tables/hanami_sql_table.h>
#include <hanami_common/logger.h>

class RequestResultTable : public HanamiSqlTable
{
   public:
    static RequestResultTable *getInstance()
    {
        if (instance == nullptr) {
            instance = new RequestResultTable();
        }
        return instance;
    }

    ~RequestResultTable();

    bool addRequestResult(json &data,
                          const UserContext &userContext,
                          Hanami::ErrorContainer &error);
    bool getRequestResult(json &result,
                          const std::string &resultUuid,
                          const UserContext &userContext,
                          Hanami::ErrorContainer &error,
                          const bool showHiddenValues);
    bool getAllRequestResult(Hanami::TableItem &result,
                             const UserContext &userContext,
                             Hanami::ErrorContainer &error);
    bool deleteRequestResult(const std::string &resultUuid,
                             const UserContext &userContext,
                             Hanami::ErrorContainer &error);

   private:
    RequestResultTable();
    static RequestResultTable *instance;
};

#endif  // HANAMI_REQUEST_RESULT_TABLE_H
