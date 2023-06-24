/**
 * @file        data_set_table.h
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

#ifndef SHIORIARCHIVE_DATA_SET_TABLE_H
#define SHIORIARCHIVE_DATA_SET_TABLE_H

#include <libKitsunemimiCommon/logger.h>
#include <database/generic_tables/hanami_sql_table.h>

namespace Kitsunemimi {
class JsonItem;
}

class DataSetTable
        : public Kitsunemimi::Hanami::HanamiSqlTable
{
public:
    DataSetTable(Kitsunemimi::Sakura::SqlDatabase* db);
    ~DataSetTable();

    bool addDataSet(Kitsunemimi::JsonItem &data,
                    const UserContext &userContext,
                    Kitsunemimi::ErrorContainer &error);
    bool getDataSet(Kitsunemimi::JsonItem &result,
                    const std::string &datasetUuid,
                    const UserContext &userContext,
                    Kitsunemimi::ErrorContainer &error,
                    const bool showHiddenValues);
    bool getAllDataSet(Kitsunemimi::TableItem &result,
                       const UserContext &userContext,
                       Kitsunemimi::ErrorContainer &error);
    bool deleteDataSet(const std::string &uuid,
                       const UserContext &userContext,
                       Kitsunemimi::ErrorContainer &error);

    bool setUploadFinish(const std::string &uuid,
                         const std::string &fileUuid,
                         Kitsunemimi::ErrorContainer &error);
};

#endif // SHIORIARCHIVE_DATA_SET_TABLE_H
