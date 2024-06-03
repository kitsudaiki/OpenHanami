#ifndef TESTTABLE_H
#define TESTTABLE_H

#include <hanami_database/sql_table.h>

namespace Hanami
{

class SqlDatabase;

class TestTable : public Hanami::SqlTable
{
   public:
    TestTable(Hanami::SqlDatabase* db);
    ~TestTable();

    bool addUser(json& data, ErrorContainer& error);
    ReturnStatus getUser(TableItem& resultTable,
                         const std::string& userID,
                         ErrorContainer& error,
                         const bool withHideValues = false);
    ReturnStatus getUser(json& resultItem,
                         const std::string& userID,
                         const bool showHiddenValues,
                         ErrorContainer& error);
    bool getAllUser(TableItem& resultItem,
                    ErrorContainer& error,
                    const bool showHiddenValues,
                    const uint64_t positionOffset = 0,
                    const uint64_t numberOfRows = 0);
    ReturnStatus deleteUser(const std::string& userID, ErrorContainer& error);
    ReturnStatus updateUser(const std::string& userID, const json& values, ErrorContainer& error);
    long getNumberOfUsers(ErrorContainer& error);
};

}  // namespace Hanami

#endif  // TESTTABLE_H
