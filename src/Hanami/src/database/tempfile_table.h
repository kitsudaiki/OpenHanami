/**
 * @file        tempfile_table.h
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

#ifndef HANAMI_CLUSTER_tempfile_TABLE_H
#define HANAMI_CLUSTER_tempfile_TABLE_H

#include <database/generic_tables/hanami_sql_table.h>
#include <hanami_common/logger.h>

class TempfileTable : public HanamiSqlTable
{
   public:
    static TempfileTable* getInstance()
    {
        if (instance == nullptr) {
            instance = new TempfileTable();
        }
        return instance;
    }

    ~TempfileTable();

    bool addTempfile(const std::string& uuid,
                     const std::string& relatedResourceType,
                     const std::string& relatedResourceUuid,
                     const std::string& name,
                     const uint64_t fileSize,
                     const std::string& location,
                     const Hanami::UserContext& userContext,
                     Hanami::ErrorContainer& error);
    bool getTempfile(json& result,
                     const std::string& tempfileUuid,
                     const Hanami::UserContext& userContext,
                     Hanami::ErrorContainer& error,
                     const bool showHiddenValues = false);
    bool getAllTempfile(Hanami::TableItem& result,
                        const Hanami::UserContext& userContext,
                        Hanami::ErrorContainer& error);
    bool deleteTempfile(const std::string& tempfileUuid,
                        const Hanami::UserContext& userContext,
                        Hanami::ErrorContainer& error);
    bool getRelatedResourceUuids(std::vector<std::string>& relatedUuids,
                                 const std::string& resourceType,
                                 const std::string& resourceUuid,
                                 const Hanami::UserContext& userContext,
                                 Hanami::ErrorContainer& error);

   private:
    TempfileTable();
    static TempfileTable* instance;
};

#endif  // HANAMI_CLUSTER_tempfile_TABLE_H
