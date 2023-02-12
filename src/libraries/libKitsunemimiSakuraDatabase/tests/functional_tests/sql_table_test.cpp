#include "sql_table_test.h"

#include <libKitsunemimiSakuraDatabase/sql_database.h>
#include <libKitsunemimiSakuraDatabase/sql_table.h>

#include <libKitsunemimiJson/json_item.h>

#include <test_table.h>

namespace Kitsunemimi::Sakura
{

SqlTable_Test::SqlTable_Test()
    : Kitsunemimi::CompareTestHelper("SqlTable_Test")
{
    initTest();

    initDatabase_test();
    initTable_test();

    create_test();
    get_test();
    getAll_test();
    update_test();
    delete_test();
    getNumberOfRows_test();
}

/**
 * @brief initTest
 */
void
SqlTable_Test::initTest()
{
    m_filePath = "/tmp/testdb.db";
    deleteFile();
}

/**
 * @brief initDB_test
 */
void
SqlTable_Test::initDatabase_test()
{
    Kitsunemimi::ErrorContainer error;
    m_db = new SqlDatabase();
    TEST_EQUAL(m_db->initDatabase(m_filePath, error), true);
}

/**
 * @brief initTable_test
 */
void
SqlTable_Test::initTable_test()
{
    m_table = new TestTable(m_db);
    ErrorContainer error;
    TEST_EQUAL(m_table->initTable(error), true);
}

/**
 * @brief create_test
 */
void
SqlTable_Test::create_test()
{
    ErrorContainer error;

    JsonItem testData;
    testData.insert("name", m_name1);
    testData.insert("pw_hash", "secret");
    testData.insert("is_admin", true);

    TEST_EQUAL(m_table->addUser(testData, error), true);


    JsonItem testData2;
    testData2.insert("name", m_name2);
    testData2.insert("pw_hash", "secret2");
    testData2.insert("is_admin", false);

    m_table->addUser(testData2, error);
}

/**
 * @brief get_test
 */
void
SqlTable_Test::get_test()
{
    JsonItem resultItem;
    TableItem resultTable;
    ErrorContainer error;

    TEST_EQUAL(m_table->getUser(resultItem, m_name1, error), true);
    TEST_EQUAL(resultItem.toString(), std::string("{\"is_admin\":true,\"name\":\"user0815\"}"));

    TEST_EQUAL(m_table->getUser(resultTable, m_name1, error), true);
    std::string compare = "+----------+----------+\n"
                          "| name     | is_admin |\n"
                          "+==========+==========+\n"
                          "| user0815 | true     |\n"
                          "+----------+----------+\n";
    TEST_EQUAL(resultTable.toString(), compare);
}

/**
 * @brief getAll_test
 */
void
SqlTable_Test::getAll_test()
{
    TableItem result;
    ErrorContainer error;

    TEST_EQUAL(m_table->getAllUser(result, error), true);
    TEST_EQUAL(result.getNumberOfRows(), 2);
    TEST_EQUAL(result.getNumberOfColums(), 2);

    result.clearTable();
    TEST_EQUAL(m_table->getAllUser(result, error, true), true);
    TEST_EQUAL(result.getNumberOfRows(), 2);
    TEST_EQUAL(result.getNumberOfColums(), 3);

    // test with limitation
    result.clearTable();
    TEST_EQUAL(m_table->getAllUser(result, error, true, 1, 10), true);
    TEST_EQUAL(result.getNumberOfRows(), 1);
    TEST_EQUAL(result.getNumberOfColums(), 3);
    TEST_EQUAL(result.getCell(0, 0), m_name2);
}

/**
 * @brief update_test
 */
void
SqlTable_Test::update_test()
{
    ErrorContainer error;

    JsonItem updateDate;
    updateDate.insert("pw_hash", "secret2");
    updateDate.insert("is_admin", false);
    TEST_EQUAL(m_table->updateUser(m_name1, updateDate, error), true);

    JsonItem resultItem;
    TableItem resultTable;

    TEST_EQUAL(m_table->getUser(resultItem, m_name1, error, true), true);
    TEST_EQUAL(resultItem.toString(),
               std::string("{\"is_admin\":false,\"name\":\"user0815\",\"pw_hash\":\"secret2\"}"));
}

/**
 * @brief delete_test
 */
void
SqlTable_Test::delete_test()
{
    ErrorContainer error;

    TEST_EQUAL(m_table->deleteUser(m_name1, error), true);
    TableItem result1;
    m_table->getAllUser(result1, error);
    TEST_EQUAL(result1.getNumberOfRows(), 1);
    TEST_EQUAL(result1.getNumberOfColums(), 2);

    TEST_EQUAL(m_table->deleteUser(m_name2, error), true);
    TableItem result2;
    m_table->getAllUser(result2, error);
    TEST_EQUAL(result2.getNumberOfRows(), 0);
    TEST_EQUAL(result2.getNumberOfColums(), 2);

    TableItem result3;
    m_table->getUser(result3, m_name1, error, true);
    TEST_EQUAL(result2.getNumberOfRows(), 0);
    TEST_EQUAL(result2.getNumberOfColums(), 2);
}

/**
 * @brief getNumberOfRows_test
 */
void
SqlTable_Test::getNumberOfRows_test()
{
    ErrorContainer error;

    TEST_EQUAL(m_table->getNumberOfUsers(error), 0);

    JsonItem testData;
    testData.insert("name", m_name1);
    testData.insert("pw_hash", "secret");
    testData.insert("is_admin", true);
    m_table->addUser(testData, error);

    TEST_EQUAL(m_table->getNumberOfUsers(error), 1);

    JsonItem testData2;
    testData2.insert("name", m_name2);
    testData2.insert("pw_hash", "secret2");
    testData2.insert("is_admin", false);
    m_table->addUser(testData2, error);

    TEST_EQUAL(m_table->getNumberOfUsers(error), 2);
}

/**
 * common usage to delete test-file
 */
void
SqlTable_Test::deleteFile()
{
    std::filesystem::path rootPathObj(m_filePath);
    if(std::filesystem::exists(rootPathObj)) {
        std::filesystem::remove(rootPathObj);
    }
}

}
