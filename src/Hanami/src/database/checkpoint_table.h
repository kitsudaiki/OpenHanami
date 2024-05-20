/**
 * @file        checkpoint_table.h
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

#ifndef HANAMI_CLUSTER_CHECKPOINT_TABLE_H
#define HANAMI_CLUSTER_CHECKPOINT_TABLE_H

#include <database/generic_tables/hanami_sql_table.h>
#include <hanami_common/logger.h>
#include <hanami_common/structs.h>

class CheckpointTable : public HanamiSqlTable
{
   public:
    static CheckpointTable* getInstance()
    {
        if (instance == nullptr) {
            instance = new CheckpointTable();
        }
        return instance;
    }

    ~CheckpointTable();

    bool addCheckpoint(json& data,
                       const Hanami::UserContext& userContext,
                       Hanami::ErrorContainer& error);
    bool getCheckpoint(json& result,
                       const std::string& checkpointUuid,
                       const Hanami::UserContext& userContext,
                       Hanami::ErrorContainer& error,
                       const bool showHiddenValues);
    bool getAllCheckpoint(Hanami::TableItem& result,
                          const Hanami::UserContext& userContext,
                          Hanami::ErrorContainer& error);
    bool deleteCheckpoint(const std::string& checkpointUuid,
                          const Hanami::UserContext& userContext,
                          Hanami::ErrorContainer& error);

   private:
    CheckpointTable();
    static CheckpointTable* instance;
};

#endif  // HANAMI_CLUSTER_CHECKPOINT_TABLE_H
