/**
 * @file       sql_table.cpp
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

#include <hanami_common/methods/string_methods.h>
#include <hanami_database/sql_database.h>
#include <hanami_database/sql_table.h>

namespace Hanami
{

/**
 * @brief constructor
 *
 * @param db pointer to database
 */
SqlTable::SqlTable(SqlDatabase* db) { m_db = db; }

/**
 * @brief destructor
 */
SqlTable::~SqlTable() {}

/**
 * @brief initalize table
 *
 * @param error reference for error-output
 *
 * @return true, if successfuly or table already exist, else false
 */
bool
SqlTable::initTable(ErrorContainer& error)
{
    return m_db->execSqlCommand(nullptr, createTableCreateQuery(), error);
}

/**
 * @brief generate markdown-text with all registered database-columns
 *
 * @param docu reference for the output of the final document
 */
void
SqlTable::createDocumentation(std::string& docu)
{
    // header
    docu.append("## " + m_tableName + "\n\n");
    docu.append("| Column-Name | Type | is primary | allow NULL|\n");
    docu.append("| --- | --- | --- | --- |\n");

    for (const DbHeaderEntry& entry : m_tableHeader) {
        docu.append("| ");

        // name
        docu.append(entry.name + " | ");

        // type
        switch (entry.type) {
            case STRING_TYPE:
                if (entry.maxLength > 0) {
                    docu.append("varchar(" + std::to_string(entry.maxLength) + ") | ");
                }
                else {
                    docu.append("text | ");
                }
                break;
            case INT_TYPE:
                docu.append("int | ");
                break;
            case BOOL_TYPE:
                docu.append("bool | ");
                break;
            case FLOAT_TYPE:
                docu.append("real | ");
                break;
        }

        // primary
        if (entry.isPrimary) {
            docu.append("true | ");
        }
        else {
            docu.append("false | ");
        }

        // NULL
        if (entry.allowNull) {
            docu.append("true | ");
        }
        else {
            docu.append("false | ");
        }

        docu.append("\n");
    }

    docu.append("\n");
}

/**
 * @brief insert values into the table
 *
 * @param values string-list with values to insert
 * @param error reference for error-output
 *
 * @return uuid of the new entry, if successful, else empty string
 */
bool
SqlTable::insertToDb(json& values, ErrorContainer& error)
{
    Hanami::TableItem resultItem;

    // get values from input to check if all required values are set
    std::vector<std::string> dbValues;
    for (const DbHeaderEntry& entry : m_tableHeader) {
        if (values.contains(entry.name) == false && entry.allowNull == false) {
            error.addMeesage("insert into dabase failed, because '" + entry.name
                             + "' is required, but missing in the input-values.");
            LOG_ERROR(error);
            return false;
        }
        if (values[entry.name].is_string()) {
            dbValues.push_back(values[entry.name]);
        }
        else {
            dbValues.push_back(values[entry.name].dump());
        }
    }

    // build and run insert-command
    if (m_db->execSqlCommand(&resultItem, createInsertQuery(dbValues), error) == false) {
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief update values within the table
 *
 * @param conditions conditions to filter table
 * @param updates json-map with key-value pairs to update
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
SqlTable::updateInDb(const std::vector<RequestCondition>& conditions,
                     const json& updates,
                     ErrorContainer& error)
{
    // precheck
    if (conditions.size() == 0) {
        error.addMeesage("no conditions given for table-access.");
        LOG_ERROR(error);
        return false;
    }

    Hanami::TableItem resultItem;
    return m_db->execSqlCommand(&resultItem, createUpdateQuery(conditions, updates), error);
}

/**
 * @brief get all rows from table
 *
 * @param resultTable pointer to table for the resuld of the query
 * @param error reference for error-output
 * @param showHiddenValues include values in output, which should normally be hidden
 * @param positionOffset offset of the rows to return
 * @param numberOfRows maximum number of results. if 0 then this value and the offset are ignored
 *
 * @return true, if successful, else false
 */
bool
SqlTable::getAllFromDb(TableItem& resultTable,
                       ErrorContainer& error,
                       const bool showHiddenValues,
                       const uint64_t positionOffset,
                       const uint64_t numberOfRows)
{
    std::vector<RequestCondition> conditions;
    if (m_db->execSqlCommand(
            &resultTable, createSelectQuery(conditions, positionOffset, numberOfRows), error)
        == false)
    {
        LOG_ERROR(error);
        return false;
    }

    // if header is missing in result, because there are no entries to list, add a default-header
    if (resultTable.getNumberOfColums() == 0) {
        for (const DbHeaderEntry& entry : m_tableHeader) {
            resultTable.addColumn(entry.name);
        }
    }

    // remove all values, which should be hide
    if (showHiddenValues == false) {
        for (const DbHeaderEntry& entry : m_tableHeader) {
            if (entry.hide) {
                resultTable.deleteColumn(entry.name);
            }
        }
    }

    return true;
}

/**
 * @brief get one or more rows from table or also the complete table
 *
 * @param resultTable pointer to table for the resuld of the query
 * @param conditions conditions to filter table
 * @param error reference for error-output
 * @param showHiddenValues include values in output, which should normally be hidden
 * @param positionOffset offset of the rows to return
 * @param numberOfRows maximum number of results. if 0 then this value and the offset are ignored
 *
 * @return true, if successful, else false
 */
bool
SqlTable::getFromDb(TableItem& resultTable,
                    const std::vector<RequestCondition>& conditions,
                    ErrorContainer& error,
                    const bool showHiddenValues,
                    const uint64_t positionOffset,
                    const uint64_t numberOfRows)
{
    if (m_db->execSqlCommand(
            &resultTable, createSelectQuery(conditions, positionOffset, numberOfRows), error)
        == false)
    {
        LOG_ERROR(error);
        return false;
    }

    // if header is missing in result, because there are no entries to list, add a default-header
    if (resultTable.getNumberOfColums() == 0) {
        for (const DbHeaderEntry& entry : m_tableHeader) {
            resultTable.addColumn(entry.name);
        }
    }

    // remove all values, which should be hide
    if (showHiddenValues == false) {
        for (const DbHeaderEntry& entry : m_tableHeader) {
            if (entry.hide) {
                resultTable.deleteColumn(entry.name);
            }
        }
    }

    return true;
}

/**
 * @brief get one or more rows from table
 *
 * @param resultTable pointer to table for the resuld of the query
 * @param conditions conditions to filter table
 * @param error reference for error-output
 * @param showHiddenValues include values in output, which should normally be hidden
 * @param positionOffset offset of the rows to return
 * @param numberOfRows maximum number of results. if 0 then this value and the offset are ignored
 *
 * @return true, if successful, else false
 */
bool
SqlTable::getFromDb(json& result,
                    const std::vector<RequestCondition>& conditions,
                    ErrorContainer& error,
                    const bool showHiddenValues,
                    const uint64_t positionOffset,
                    const uint64_t numberOfRows)
{
    // precheck
    if (conditions.size() == 0) {
        error.addMeesage("no conditions given for table-access.");
        LOG_ERROR(error);
        return false;
    }

    // run select-query
    TableItem tableResult;
    if (m_db->execSqlCommand(
            &tableResult, createSelectQuery(conditions, positionOffset, numberOfRows), error)
        == false)
    {
        LOG_ERROR(error);
        return false;
    }

    if (tableResult.getNumberOfRows() == 0) {
        return true;
    }

    // convert table-row to json
    processGetResult(result, tableResult);

    // remove all values, which should be hide
    if (showHiddenValues == false) {
        for (const DbHeaderEntry& entry : m_tableHeader) {
            if (entry.hide) {
                result.erase(entry.name);
            }
        }
    }

    return true;
}

/**
 * @brief Request number of rows of the database-table
 *
 * @param error reference for error-output
 *
 * @return -1 if request against database failed, else number of rows
 */
long
SqlTable::getNumberOfRows(ErrorContainer& error)
{
    Hanami::TableItem resultItem;
    if (m_db->execSqlCommand(&resultItem, createCountQuery(), error) == false) {
        return -1;
    }

    return resultItem.getBody()[0][0];
}

/**
 * @brief delete all entries for the table
 *
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
SqlTable::deleteAllFromDb(ErrorContainer& error)
{
    const std::vector<RequestCondition> conditions;
    Hanami::TableItem resultItem;
    return m_db->execSqlCommand(&resultItem, createDeleteQuery(conditions), error);
}

/**
 * @brief delete one of more rows from database
 *
 * @param conditions conditions to filter table
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
SqlTable::deleteFromDb(const std::vector<RequestCondition>& conditions, ErrorContainer& error)
{
    // precheck
    if (conditions.size() == 0) {
        error.addMeesage("no conditions given for table-access.");
        LOG_ERROR(error);
        return false;
    }

    Hanami::TableItem resultItem;
    return m_db->execSqlCommand(&resultItem, createDeleteQuery(conditions), error);
}

/**
 * @brief create a sql-query to create a table
 *
 * @return created sql-query
 */
const std::string
SqlTable::createTableCreateQuery()
{
    std::string command = "CREATE TABLE IF NOT EXISTS ";
    command.append(m_tableName);
    command.append(" (");

    // create all field of the table
    for (uint32_t i = 0; i < m_tableHeader.size(); i++) {
        const DbHeaderEntry* entry = &m_tableHeader[i];
        if (i != 0) {
            command.append(" , ");
        }
        command.append(entry->name + "  ");

        // set type of the value
        switch (entry->type) {
            case STRING_TYPE:
                if (entry->maxLength > 0) {
                    command.append("varchar(");
                    command.append(std::to_string(entry->maxLength));
                    command.append(") ");
                }
                else {
                    command.append("text ");
                }
                break;
            case INT_TYPE:
                command.append("int ");
                break;
            case BOOL_TYPE:
                command.append("bool ");
                break;
            case FLOAT_TYPE:
                command.append("real ");
                break;
        }

        // set if key is primary key
        if (entry->isPrimary) {
            command.append("PRIMARY KEY ");
        }

        // set if value is not allowed to be null
        if (entry->allowNull == false) {
            command.append("NOT NULL ");
        }
    }

    command.append(");");

    return command;
}

/**
 * @brief create a sql-query to get a line from the table
 *
 * @param conditions conditions to filter table
 * @param positionOffset offset of the rows to return
 * @param numberOfRows maximum number of results. if 0 then this value and the offset are ignored
 *
 * @return created sql-query
 */
const std::string
SqlTable::createSelectQuery(const std::vector<RequestCondition>& conditions,
                            const uint64_t positionOffset,
                            const uint64_t numberOfRows)
{
    std::string command = "SELECT * from " + m_tableName;

    // filter
    if (conditions.size() > 0) {
        command.append(" WHERE ");

        for (uint32_t i = 0; i < conditions.size(); i++) {
            if (i > 0) {
                command.append(" AND ");
            }
            const RequestCondition* condition = &conditions.at(i);
            command.append(condition->colName);
            command.append("='");
            command.append(condition->value);
            command.append("' ");
        }
    }

    // limit number of results
    if (numberOfRows > 0) {
        command.append(" LIMIT ");
        command.append(std::to_string(numberOfRows));
        command.append(" OFFSET ");
        command.append(std::to_string(positionOffset));
    }

    command.append(" ;");

    return command;
}

/**
 * @brief create a sql-query to update values within the table
 *
 * @param conditions conditions to filter table
 * @param updates json-map with key-value pairs to update
 *
 * @return created sql-query
 */
const std::string
SqlTable::createUpdateQuery(const std::vector<RequestCondition>& conditions, const json& updates)
{
    std::string command = "UPDATE ";
    command.append(m_tableName);

    // add set-section
    command.append(" SET ");
    std::vector<std::string> keys;
    for (const auto& [key, _] : updates.items()) {
        keys.push_back(key);
    }
    for (uint32_t i = 0; i < keys.size(); i++) {
        if (i > 0) {
            command.append(" , ");
        }
        command.append(keys.at(i));
        if (updates[keys.at(i)].is_string()) {
            command.append("='" + std::string(updates[keys.at(i)]) + "' ");
        }
        else {
            command.append("='" + updates[keys.at(i)].dump() + "' ");
        }
    }

    // add where-section
    if (conditions.size() > 0) {
        command.append(" WHERE ");

        for (uint32_t i = 0; i < conditions.size(); i++) {
            if (i > 0) {
                command.append(" AND ");
            }
            const RequestCondition* condition = &conditions.at(i);
            command.append(condition->colName);
            command.append("='");
            command.append(condition->value);
            command.append("' ");
        }
    }
    command.append(" ;");

    return command;
}

/**
 * @brief create a sql-query to insert values into the table
 *
 * @param values list of values to insert
 *
 * @return created sql-query
 */
const std::string
SqlTable::createInsertQuery(const std::vector<std::string>& values)
{
    std::string command = "INSERT INTO ";
    command.append(m_tableName);
    command.append("(");

    // create fields
    for (uint32_t i = 0; i < m_tableHeader.size(); i++) {
        const DbHeaderEntry* entry = &m_tableHeader[i];
        if (i != 0) {
            command.append(" , ");
        }
        command.append(entry->name);
    }

    // create values
    command.append(") VALUES (");

    for (uint32_t i = 0; i < m_tableHeader.size(); i++) {
        if (i != 0) {
            command.append(" , ");
        }
        command.append("'");
        command.append(values.at(i));
        command.append("'");
    }

    command.append(" );");

    return command;
}

/**
 * @brief create query to delete rows from table
 *
 * @param conditions conditions to filter table
 *
 * @return created sql-query
 */
const std::string
SqlTable::createDeleteQuery(const std::vector<RequestCondition>& conditions)
{
    std::string command = "DELETE FROM ";
    command.append(m_tableName);

    if (conditions.size() > 0) {
        command.append(" WHERE ");

        for (uint32_t i = 0; i < conditions.size(); i++) {
            if (i > 0) {
                command.append(" AND ");
            }
            const RequestCondition* condition = &conditions.at(i);
            command.append(condition->colName);
            command.append("='");
            command.append(condition->value);
            command.append("' ");
        }
    }
    command.append(" ;");

    return command;
}

/**
 * @brief create a sql-query to request number of rows of the table
 *
 * @return created sql-query
 */
const std::string
SqlTable::createCountQuery()
{
    std::string command = "SELECT COUNT(*) as number_of_rows FROM ";
    command.append(m_tableName);
    command.append(";");

    return command;
}

/**
 * @brief convert first row together with header into json
 *
 * @param result reference for json-formated output
 * @param tableContent table-input with at least one row
 */
void
SqlTable::processGetResult(json& result, TableItem& tableContent)
{
    if (tableContent.getNumberOfRows() == 0) {
        return;
    }

    // prepare result
    const json firstRow = tableContent.getBody()[0];

    for (uint32_t i = 0; i < m_tableHeader.size(); i++) {
        result[m_tableHeader.at(i).name] = firstRow[i];
    }
}

}  // namespace Hanami
