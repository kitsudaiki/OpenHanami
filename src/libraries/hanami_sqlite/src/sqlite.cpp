/**
 *  @file    sqlite.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 *
 *  @brief simple class for easier handling of sqlite-database
 *
 *  @detail This class provides only three abilities: open and close sqlite databases and
 *          execute sql-commands. The results of the database request are converted into
 *          table-itmes of hanami_common.
 *
 *          This class was created with the help of:
 *              https://www.tutorialspoint.com/sqlite/sqlite_c_cpp.htm
 */

#include <hanami_sqlite/sqlite.h>
#include <hanami_common/items/table_item.h>
#include <hanami_common/items/data_items.h>
#include <hanami_json/json_item.h>

using Hanami::DataItem;
using Hanami::DataMap;
using Hanami::DataArray;
using Hanami::DataValue;

namespace Hanami
{

/**
 * @brief constructor
 */
Sqlite::Sqlite() {}

/**
 * @brief destcutor
 */
Sqlite::~Sqlite()
{
    closeDB();
}

/**
 * @brief initialize database
 *
 * @param path file-path of the existing or new sqlite file
 * @param error reference for error-message output
 *
 * @return true, if seccessful, else false
 */
bool
Sqlite::initDB(const std::string &path,
               ErrorContainer &error)
{
    m_rc = sqlite3_open(path.c_str(), &m_db);

    if(m_rc != 0)
    {
        error.addMeesage("Can't open database: \n" + std::string(sqlite3_errmsg(m_db)));
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief callback-fucntion, which is called for every row of the result of the sql-request
 *
 * @param data Data provided in the 4th argument of sqlite3_exec()
 * @param argc The number of columns in row
 * @param argv An array of strings representing fields in the row
 * @param azColName An array of strings representing column names
 *
 * @return 0
 */
static int
callback(void* data,
         int argc,
         char** argv,
         char** azColName)
{
    // precheck
    if(data == nullptr) {
        return 0;
    }

    ErrorContainer error;
    TableItem* result = static_cast<TableItem*>(data);

    // add columns to the table-item, but only the first time
    // because this callback is called for every row
    if(result->getNumberOfColums() == 0)
    {
        for(int i = 0; i < argc; i++)
        {
            const std::string columnName = azColName[i];
            result->addColumn(columnName);
        }
    }

    const std::regex intVal("^-?([0-9]+)$");
    const std::regex floatVal("^-?([0-9]+)\\.([0-9]+)$");

    // collect row-data
    DataArray* row = new DataArray();
    for(int i = 0; i < argc; i++)
    {
        if(argv[i])
        {
            const std::string value = argv[i];

            // true
            if(value == "True"
                    || value == "true"
                    || value == "TRUE")
            {
                row->append(new DataValue(true));
            }
            // false
            else if(value == "False"
                    || value == "false"
                    || value == "FALSE")
            {
                row->append(new DataValue(false));
            }
            // int/long
            else if(regex_match(value, intVal))
            {
                const long longVal = std::strtol(value.c_str(), NULL, 10);
                row->append(new DataValue(longVal));
            }
            // float/double
            else if(regex_match(value, floatVal))
            {
                const double doubleVal = std::strtod(value.c_str(), NULL);
                row->append(new DataValue(doubleVal));
            }
            // json-map
            else if(value.size() > 0
                    && value.at(0) == '{')
            {
                Hanami::JsonItem json;
                if(json.parse(value, error) == false) {
                    row->append(new DataValue(value));
                }

                row->append(json.stealItemContent());
            }
            // json-array
            else if(value.size() > 0
                    && value.at(0) == '[')
            {
                Hanami::JsonItem json;
                if(json.parse(value, error) == false) {
                    row->append(new DataValue(value));
                }

                row->append(json.stealItemContent());
            }
            // string
            else
            {
                row->append(new DataValue(value));
            }
        }
        else
        {
            row->append(new DataValue("NULL"));
        }
    }

    // add row to the table-item
    result->addRow(row);

    return 0;
}

/**
 * @brief execute a sql-command on the sqlite database
 *
 * @param resultTable table-item, which should contain the result
 * @param command sql-command for execution
 * @param error reference for error-message output
 *
 * @return true, if seccessful, else false
 */
bool
Sqlite::execSqlCommand(TableItem* resultTable,
                       const std::string &command,
                       ErrorContainer &error)
{
    // run command
    char* err = nullptr;
    m_rc = sqlite3_exec(m_db,
                        command.c_str(),
                        callback,
                        static_cast<void*>(resultTable),
                        &err);

    // check result state
    if(m_rc != SQLITE_OK)
    {
        error.addMeesage("SQL error: " + std::string(err));
        sqlite3_free(err);
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief close the database
 *
 * @return false if db was not open, else true
 */
bool
Sqlite::closeDB()
{
    if(m_db != nullptr)
    {
        sqlite3_close(m_db);
        m_db = nullptr;
        return true;
    }

    return false;
}

}
