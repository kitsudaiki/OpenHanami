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
    static RequestResultTable* getInstance()
    {
        if (instance == nullptr) {
            instance = new RequestResultTable();
        }
        return instance;
    }

    struct ResultDbEntry {
        std::string uuid = "";
        std::string projectId = "";
        std::string ownerId = "";
        std::string visibility = "";
        std::string name = "";
        json data;
    };

    ~RequestResultTable();

    ReturnStatus addRequestResult(ResultDbEntry& resultData,
                                  const Hanami::UserContext& userContext,
                                  Hanami::ErrorContainer& error);
    ReturnStatus getRequestResult(ResultDbEntry& result,
                                  const std::string& resultUuid,
                                  const Hanami::UserContext& userContext,
                                  Hanami::ErrorContainer& error);
    ReturnStatus getRequestResult(json& result,
                                  const std::string& resultUuid,
                                  const Hanami::UserContext& userContext,
                                  const bool showHiddenValues,
                                  Hanami::ErrorContainer& error);
    bool getAllRequestResult(Hanami::TableItem& result,
                             const Hanami::UserContext& userContext,
                             Hanami::ErrorContainer& error);
    ReturnStatus deleteRequestResult(const std::string& resultUuid,
                                     const Hanami::UserContext& userContext,
                                     Hanami::ErrorContainer& error);

   private:
    RequestResultTable();
    static RequestResultTable* instance;
};

#endif  // HANAMI_REQUEST_RESULT_TABLE_H
