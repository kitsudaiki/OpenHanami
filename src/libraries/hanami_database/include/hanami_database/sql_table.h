/**
 * @file       sql_table.h
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

#ifndef KITSUNEMIMI_SAKURA_DATABASE_SQL_TABLE_H
#define KITSUNEMIMI_SAKURA_DATABASE_SQL_TABLE_H

#include <vector>
#include <string>
#include <uuid/uuid.h>

#include <hanami_common/items/data_items.h>
#include <hanami_common/logger.h>

namespace Kitsunemimi {
class JsonItem;
}

namespace Kitsunemimi::Sakura
{
class SqlDatabase;

class SqlTable
{
public:
    SqlTable(SqlDatabase* db);
    virtual ~SqlTable();

    bool initTable(ErrorContainer &error);
    void createDocumentation(std::string &docu);

protected:
    enum DbVataValueTypes
    {
        STRING_TYPE = 0,
        INT_TYPE = 1,
        BOOL_TYPE = 2,
        FLOAT_TYPE = 3
    };

    struct DbHeaderEntry
    {
        std::string name = "";
        int maxLength = -1;
        DbVataValueTypes type = STRING_TYPE;
        bool isPrimary = false;
        bool allowNull = false;
        bool hide = false;
    };

    struct RequestCondition
    {
        std::string colName;
        std::string value;

        RequestCondition(const std::string &colName,
                         const std::string &value)
        {
            this->colName = colName;
            this->value = value;
        }
    };

    std::vector<DbHeaderEntry> m_tableHeader;
    std::string m_tableName = "";

    bool insertToDb(JsonItem &values,
                    ErrorContainer &error);
    bool updateInDb(const std::vector<RequestCondition> &conditions,
                    const JsonItem &updates,
                    ErrorContainer &error);
    bool getAllFromDb(TableItem &resultTable,
                      ErrorContainer &error,
                      const bool showHiddenValues = false,
                      const uint64_t positionOffset = 0,
                      const uint64_t numberOfRows = 0);
    bool getFromDb(TableItem &resultTable,
                   const std::vector<RequestCondition> &conditions,
                   ErrorContainer &error,
                   const bool showHiddenValues = false,
                   const uint64_t positionOffset = 0,
                   const uint64_t numberOfRows = 0);
    bool getFromDb(JsonItem &result,
                   const std::vector<RequestCondition> &conditions,
                   ErrorContainer &error,
                   const bool showHiddenValues = false,
                   const uint64_t positionOffset = 0,
                   const uint64_t numberOfRows = 0);
    long getNumberOfRows(ErrorContainer &error);
    bool deleteAllFromDb(ErrorContainer &error);
    bool deleteFromDb(const std::vector<RequestCondition> &conditions,
                      ErrorContainer &error);
private:
    SqlDatabase* m_db = nullptr;

    const std::string createTableCreateQuery();
    const std::string createSelectQuery(const std::vector<RequestCondition> &conditions,
                                        const uint64_t positionOffset,
                                        const uint64_t numberOfRows);
    const std::string createUpdateQuery(const std::vector<RequestCondition> &conditions,
                                        const JsonItem &updates);
    const std::string createInsertQuery(const std::vector<std::string> &values);
    const std::string createDeleteQuery(const std::vector<RequestCondition> &conditions);
    const std::string createCountQuery();

    void processGetResult(JsonItem &result,
                          TableItem &tableContent);
};

}

#endif // KITSUNEMIMI_SAKURA_DATABASE_SQL_TABLE_H
