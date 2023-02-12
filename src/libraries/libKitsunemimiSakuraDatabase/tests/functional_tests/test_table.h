#ifndef TESTTABLE_H
#define TESTTABLE_H

#include <libKitsunemimiSakuraDatabase/sql_table.h>

namespace Kitsunemimi::Sakura
{

class SqlDatabase;

class TestTable :
        public Kitsunemimi::Sakura::SqlTable
{
public:
    TestTable(Kitsunemimi::Sakura::SqlDatabase* db);
    ~TestTable();

    bool addUser(JsonItem &data,
                 ErrorContainer &error);
    bool getUser(TableItem &resultTable,
                 const std::string &userID,
                 ErrorContainer &error,
                 const bool withHideValues = false);
    bool getUser(JsonItem &resultItem,
                 const std::string &userID,
                 ErrorContainer &error,
                 const bool showHiddenValues = false);
    bool getAllUser(TableItem &resultItem,
                    ErrorContainer &error,
                    const bool showHiddenValues = false,
                    const uint64_t positionOffset = 0,
                    const uint64_t numberOfRows = 0);
    bool deleteUser(const std::string &userID,
                    ErrorContainer &error);
    bool updateUser(const std::string &userID,
                    const JsonItem &values,
                    ErrorContainer &error);
    long getNumberOfUsers(ErrorContainer &error);
};

}

#endif // TESTTABLE_H
