#ifndef TESTTABLE_H
#define TESTTABLE_H

#include <hanami_database/sql_table.h>

namespace Hanami
{

class SqlDatabase;

class TestTable :
        public Hanami::SqlTable
{
public:
    TestTable(Hanami::SqlDatabase* db);
    ~TestTable();

    bool addUser(json &data,
                 ErrorContainer &error);
    bool getUser(TableItem &resultTable,
                 const std::string &userID,
                 ErrorContainer &error,
                 const bool withHideValues = false);
    bool getUser(json &resultItem,
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
                    const json &values,
                    ErrorContainer &error);
    long getNumberOfUsers(ErrorContainer &error);
};

}

#endif // TESTTABLE_H
