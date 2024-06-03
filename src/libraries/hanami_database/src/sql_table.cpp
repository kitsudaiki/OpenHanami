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

#include <hanami_common/functions/string_functions.h>
#include <hanami_common/functions/time_functions.h>
#include <hanami_database/sql_database.h>
#include <hanami_database/sql_table.h>

namespace Hanami
{

/**
 * @brief constructor
 *
 * @param db pointer to database
 */
SqlTable::SqlTable(SqlDatabase* db)
{
    m_db = db;

    registerColumn("status", STRING_TYPE).setMaxLength(10);

    registerColumn("deleted_at", STRING_TYPE).setMaxLength(64);

    registerColumn("created_at", STRING_TYPE).setMaxLength(64);
}

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
            case HASH_TYPE:
                docu.append("hash | ");
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
 * @brief get number of columns of the database-table
 */
uint64_t
SqlTable::getNumberOfColumns() const
{
    return m_tableHeader.size() - 2;
}

/**
 * @brief register a new column in the table
 *
 * @param name name of the colume
 * @param type type of the column
 *
 * @return reference to new entry
 */
SqlTable::DbHeaderEntry&
SqlTable::registerColumn(const std::string& name, const DbVataValueTypes type)
{
    DbHeaderEntry newEntry;
    newEntry.name = name;
    newEntry.type = type;
    m_tableHeader.push_back(newEntry);

    return m_tableHeader[m_tableHeader.size() - 1];
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

    values["created_at"] = Hanami::getDatetime();
    values["status"] = "active";
    values["deleted_at"] = "";

    // get values from input to check if all required values are set
    std::vector<std::string> dbValues;
    for (const DbHeaderEntry& entry : m_tableHeader) {
        if (values.contains(entry.name) == false && entry.allowNull == false) {
            error.addMessage("insert into dabase failed, because '" + entry.name
                             + "' is required, but missing in the input-values.");
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
ReturnStatus
SqlTable::updateInDb(std::vector<RequestCondition>& conditions,
                     const json& updates,
                     ErrorContainer& error)
{
    if (conditions.size() != 0) {
        // precheck
        json getResult;
        const ReturnStatus ret = getFromDb(getResult, conditions, false, true, error);
        if (ret != OK) {
            return ret;
        }
    }

    Hanami::TableItem resultItem;
    if (m_db->execSqlCommand(&resultItem, createUpdateQuery(conditions, updates), error) == false) {
        return ERROR;
    }

    return OK;
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
    if (getFromDb(
            resultTable, conditions, showHiddenValues, false, error, positionOffset, numberOfRows)
        == ERROR)
    {
        return false;
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
ReturnStatus
SqlTable::getFromDb(TableItem& resultTable,
                    std::vector<RequestCondition>& conditions,
                    const bool showHiddenValues,
                    const bool expectAtLeastOne,
                    ErrorContainer& error,
                    const uint64_t positionOffset,
                    const uint64_t numberOfRows)
{
    conditions.emplace_back("status", "active");

    if (m_db->execSqlCommand(
            &resultTable,
            createSelectQuery(conditions, showHiddenValues, positionOffset, numberOfRows),
            error)
        == false)
    {
        LOG_ERROR(error);
        return ERROR;
    }

    // if header is missing in result, because there are no entries to list, add a default-header
    if (resultTable.getNumberOfColums() == 0) {
        if (expectAtLeastOne) {
            return INVALID_INPUT;
        }
        for (const DbHeaderEntry& entry : m_tableHeader) {
            if (entry.name == "status" || entry.name == "deleted_at") {
                continue;
            }
            if (showHiddenValues || entry.hide == false) {
                resultTable.addColumn(entry.name);
            }
        }
    }

    return OK;
}

/**
 * @brief get one or more rows from table
 *
 * @param resultTable pointer to table for the resuld of the query
 * @param conditions conditions to filter table
 * @param error reference for error-output
 * @param showHiddenValues include values in output, which should normally be hidden
 * @param expectAtLeastOne if false, there is no return false, if the db doesn't return any results
 * @param positionOffset offset of the rows to return
 * @param numberOfRows maximum number of results. if 0 then this value and the offset are ignored
 *
 * @return OK if found, INVALID_INPUT if not found, ERROR in case of internal error
 */
ReturnStatus
SqlTable::getFromDb(json& result,
                    std::vector<RequestCondition>& conditions,
                    const bool showHiddenValues,
                    const bool expectAtLeastOne,
                    ErrorContainer& error,
                    const uint64_t positionOffset,
                    const uint64_t numberOfRows)
{
    // precheck
    if (conditions.size() == 0) {
        error.addMessage("no conditions given for table-access.");
        LOG_ERROR(error);
        return INVALID_INPUT;
    }

    TableItem resultTable;
    const ReturnStatus ret = getFromDb(resultTable,
                                       conditions,
                                       showHiddenValues,
                                       expectAtLeastOne,
                                       error,
                                       positionOffset,
                                       numberOfRows);
    if (ret != OK) {
        return ret;
    }

    // convert table-row to json
    processGetResult(result, resultTable, showHiddenValues);

    return OK;
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
    std::vector<RequestCondition> conditions;
    return deleteFromDb(conditions, error);
}

/**
 * @brief delete one of more rows from database
 *
 * @param conditions conditions to filter table
 * @param error reference for error-output
 *
 * @return OK if found, INVALID_INPUT if not found, ERROR in case of internal error
 */
ReturnStatus
SqlTable::deleteFromDb(std::vector<RequestCondition>& conditions, ErrorContainer& error)
{
    json update;
    update["status"] = "deleted";
    update["deleted_at"] = getDatetime();

    return updateInDb(conditions, update, error);
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
            case HASH_TYPE:
                command.append("text ");
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
                            const bool showHiddenValues,
                            const uint64_t positionOffset,
                            const uint64_t numberOfRows)
{
    // add select
    std::string command = "SELECT ";
    // offset of 2, because status and deleted_at are removed
    for (uint64_t i = 2; i < m_tableHeader.size(); i++) {
        if (showHiddenValues || m_tableHeader.at(i).hide == false) {
            if (i != 2) {
                command.append(", ");
            }

            command.append(m_tableHeader.at(i).name);
        }
    }

    // add from
    command.append(" FROM ");
    command.append(m_tableName);

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
 * @brief create a sql-query to request number of rows of the table
 *
 * @return created sql-query
 */
const std::string
SqlTable::createCountQuery()
{
    std::string command = "SELECT COUNT(*) as number_of_rows FROM ";
    command.append(m_tableName);
    command.append(" where status='active';");

    return command;
}

/**
 * @brief convert first row together with header into json
 *
 * @param result reference for json-formated output
 * @param tableContent table-input with at least one row
 */
void
SqlTable::processGetResult(json& result, TableItem& tableContent, const bool showHiddenValues)
{
    if (tableContent.getNumberOfRows() == 0) {
        return;
    }

    // prepare result
    const json firstRow = tableContent.getBody()[0];

    // offset of 2, because status and deleted_at are removed from result
    uint32_t pos = 0;
    for (uint32_t i = 2; i < m_tableHeader.size(); i++) {
        const DbHeaderEntry& entry = m_tableHeader.at(i);
        if (entry.hide == false || showHiddenValues == true) {
            result[entry.name] = firstRow[pos];
            pos++;
        }
    }
}

}  // namespace Hanami
