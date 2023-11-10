#include "test_table.h"

#include <hanami_database/sql_database.h>
#include <hanami_database/sql_table.h>

namespace Hanami
{

TestTable::TestTable(Hanami::SqlDatabase* db) : SqlTable(db)
{
    m_tableName = "users";

    DbHeaderEntry userName;
    userName.name = "name";
    userName.maxLength = 256;
    m_tableHeader.push_back(userName);

    DbHeaderEntry pwHash;
    pwHash.name = "pw_hash";
    pwHash.maxLength = 64;
    pwHash.hide = true;
    m_tableHeader.push_back(pwHash);

    DbHeaderEntry isAdmin;
    isAdmin.name = "is_admin";
    isAdmin.type = BOOL_TYPE;
    m_tableHeader.push_back(isAdmin);
}

TestTable::~TestTable() {}

/**
 * @brief addUser
 */
bool
TestTable::addUser(json& data, ErrorContainer& error)
{
    return insertToDb(data, error);
}

/**
 * @brief getUser
 */
bool
TestTable::getUser(TableItem& resultTable,
                   const std::string& userID,
                   ErrorContainer& error,
                   const bool withHideValues)
{
    std::vector<RequestCondition> conditions;
    conditions.emplace_back("name", userID);
    return getFromDb(resultTable, conditions, error, withHideValues);
}

/**
 * @brief getUser
 */
bool
TestTable::getUser(json& resultItem,
                   const std::string& userID,
                   ErrorContainer& error,
                   const bool withHideValues)
{
    std::vector<RequestCondition> conditions;
    conditions.emplace_back("name", userID);
    return getFromDb(resultItem, conditions, error, withHideValues);
}

/**
 * @brief updateUser
 */
bool
TestTable::updateUser(const std::string& userID, const json& values, ErrorContainer& error)
{
    std::vector<RequestCondition> conditions;
    conditions.emplace_back("name", userID);
    return updateInDb(conditions, values, error);
}

/**
 * @brief getNumberOfUsers
 */
long
TestTable::getNumberOfUsers(ErrorContainer& error)
{
    return getNumberOfRows(error);
}

/**
 * @brief getAllUser
 */
bool
TestTable::getAllUser(TableItem& resultItem,
                      ErrorContainer& error,
                      const bool showHiddenValues,
                      const uint64_t positionOffset,
                      const uint64_t numberOfRows)
{
    return getAllFromDb(resultItem, error, showHiddenValues, positionOffset, numberOfRows);
}

/**
 * @brief deleteUser
 */
bool
TestTable::deleteUser(const std::string& userID, ErrorContainer& error)
{
    std::vector<RequestCondition> conditions;
    conditions.emplace_back("name", userID);
    return deleteFromDb(conditions, error);
}

}  // namespace Hanami
