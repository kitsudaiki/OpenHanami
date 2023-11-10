/**
 *  @file    sqlite.h
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

#ifndef SQLITE_H
#define SQLITE_H

#include <hanami_common/logger.h>
#include <sqlite3.h>

#include <iostream>
#include <nlohmann/json.hpp>
#include <regex>
#include <utility>
#include <vector>

using json = nlohmann::json;

namespace Hanami
{
class TableItem;

class Sqlite
{
   public:
    Sqlite();
    ~Sqlite();

    bool initDB(const std::string& path, ErrorContainer& error);

    bool execSqlCommand(TableItem* resultTable, const std::string& command, ErrorContainer& error);

    bool closeDB();

   private:
    sqlite3* m_db = nullptr;
    int m_rc = 0;
};

}  // namespace Hanami

#endif  // SQLITE_H
