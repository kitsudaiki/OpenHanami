/**
 *  @file    sqlite_test.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include "sqlite_test.h"

#include <hanami_sqlite/sqlite.h>
#include <hanami_common/items/table_item.h>

namespace Kitsunemimi
{

Sqlite_Test::Sqlite_Test()
    : Kitsunemimi::CompareTestHelper("Sqlite_Test")
{
    initTest();
    initDB_test();
    execSqlCommand_test();
    closeDB_test();
    closeTest();
}

/**
 * @brief initTest
 */
void
Sqlite_Test::initTest()
{
    m_filePath = "/tmp/testdb.db";
    deleteFile();
}

/**
 * @brief initDB_test
 */
void
Sqlite_Test::initDB_test()
{
    Sqlite testDB;

    ErrorContainer error;

    TEST_EQUAL(testDB.initDB(m_filePath, error), true);

    deleteFile();
}

/**
 * @brief execSqlCommand_test
 */
void
Sqlite_Test::execSqlCommand_test()
{
    Sqlite testDB;
    ErrorContainer error;
    testDB.initDB(m_filePath, error);

    Kitsunemimi::TableItem resultItem;

    //-----------------------------------------------------------------
    // CREATE TABLE
    //-----------------------------------------------------------------
    std::string sql = "CREATE TABLE COMPANY("
                      "ID INT PRIMARY KEY     NOT NULL,"
                      "NAME           TEXT    NOT NULL,"
                      "AGE            INT     NOT NULL,"
                      "ADDRESS        CHAR(50),"
                      "SALARY         REAL );";

    TEST_EQUAL(testDB.execSqlCommand(nullptr, sql, error), true);

    //-----------------------------------------------------------------
    // INSERT
    //-----------------------------------------------------------------
    sql = "INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) "
          "VALUES (1, 'Paul', 32, '{country: \"California\"}', 20000.00 ); "
          "INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) "
          "VALUES (2, 'Allen', 25, '{country: \"Texas\"}', 15000.00 ); "
          "INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY)"
          "VALUES (3, 'Teddy', 23, '{country: \"Norway\"}', 20000.00 );"
          "INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY)"
          "VALUES (4, 'Mark', 25, '{country: \"Rich-Mond\"}', 65000.00 );";

    TEST_EQUAL(testDB.execSqlCommand(nullptr, sql, error), true);

    //-----------------------------------------------------------------
    // SELECT
    //-----------------------------------------------------------------
    sql = "SELECT * from COMPANY";

    resultItem.clearTable();
    TEST_EQUAL(testDB.execSqlCommand(&resultItem, sql, error), true);

    std::string compare = "+----+-------+-----+--------------------------+--------------+\n"
                          "| ID | NAME  | AGE | ADDRESS                  | SALARY       |\n"
                          "+====+=======+=====+==========================+==============+\n"
                          "| 1  | Paul  | 32  | {\"country\":\"California\"} | 20000.000000 |\n"
                          "+----+-------+-----+--------------------------+--------------+\n"
                          "| 2  | Allen | 25  | {\"country\":\"Texas\"}      | 15000.000000 |\n"
                          "+----+-------+-----+--------------------------+--------------+\n"
                          "| 3  | Teddy | 23  | {\"country\":\"Norway\"}     | 20000.000000 |\n"
                          "+----+-------+-----+--------------------------+--------------+\n"
                          "| 4  | Mark  | 25  | {\"country\":\"Rich-Mond\"}  | 65000.000000 |\n"
                          "+----+-------+-----+--------------------------+--------------+\n";
    TEST_EQUAL(resultItem.toString(), compare);

    //-----------------------------------------------------------------
    // UPDATE
    //-----------------------------------------------------------------
    sql = "UPDATE COMPANY set SALARY = 25000.00 where ID=1; "
          "SELECT * from COMPANY";

    resultItem.clearTable();
    TEST_EQUAL(testDB.execSqlCommand(&resultItem, sql, error), true);

    compare = "+----+-------+-----+--------------------------+--------------+\n"
              "| ID | NAME  | AGE | ADDRESS                  | SALARY       |\n"
              "+====+=======+=====+==========================+==============+\n"
              "| 1  | Paul  | 32  | {\"country\":\"California\"} | 25000.000000 |\n"
              "+----+-------+-----+--------------------------+--------------+\n"
              "| 2  | Allen | 25  | {\"country\":\"Texas\"}      | 15000.000000 |\n"
              "+----+-------+-----+--------------------------+--------------+\n"
              "| 3  | Teddy | 23  | {\"country\":\"Norway\"}     | 20000.000000 |\n"
              "+----+-------+-----+--------------------------+--------------+\n"
              "| 4  | Mark  | 25  | {\"country\":\"Rich-Mond\"}  | 65000.000000 |\n"
              "+----+-------+-----+--------------------------+--------------+\n";
    TEST_EQUAL(resultItem.toString(), compare);

    //-----------------------------------------------------------------
    // DELETE
    //-----------------------------------------------------------------
    sql = "DELETE from COMPANY where ID=2; "
          "SELECT * from COMPANY";

    resultItem.clearTable();
    TEST_EQUAL(testDB.execSqlCommand(&resultItem, sql, error), true);

    compare = "+----+-------+-----+--------------------------+--------------+\n"
              "| ID | NAME  | AGE | ADDRESS                  | SALARY       |\n"
              "+====+=======+=====+==========================+==============+\n"
              "| 1  | Paul  | 32  | {\"country\":\"California\"} | 25000.000000 |\n"
              "+----+-------+-----+--------------------------+--------------+\n"
              "| 3  | Teddy | 23  | {\"country\":\"Norway\"}     | 20000.000000 |\n"
              "+----+-------+-----+--------------------------+--------------+\n"
              "| 4  | Mark  | 25  | {\"country\":\"Rich-Mond\"}  | 65000.000000 |\n"
              "+----+-------+-----+--------------------------+--------------+\n";
    TEST_EQUAL(resultItem.toString(), compare);


    testDB.closeDB();

    deleteFile();
}

/**
 * @brief closeDB_test
 */
void
Sqlite_Test::closeDB_test()
{
    Sqlite testDB;

    TEST_EQUAL(testDB.closeDB(), false);

    ErrorContainer error;
    testDB.initDB(m_filePath, error);

    TEST_EQUAL(testDB.closeDB(), true);
    TEST_EQUAL(testDB.closeDB(), false);

    deleteFile();
}

/**
 * @brief closeTest
 */
void
Sqlite_Test::closeTest()
{
    deleteFile();
}

/**
 * common usage to delete test-file
 */
void
Sqlite_Test::deleteFile()
{
    std::filesystem::path rootPathObj(m_filePath);
    if(std::filesystem::exists(rootPathObj)) {
        std::filesystem::remove(rootPathObj);
    }
}

}
