/**
 * @file        requestResult_table_test.cpp
 *
 * @author      Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright   Apache License Version 2.0
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

#include "request_result_table_test.h"

#include <hanami_common/functions/file_functions.h>
#include <hanami_common/logger.h>
#include <hanami_common/uuid.h>
#include <hanami_database/sql_database.h>
#include <src/database/request_result_table.h>

RequestResultTable_Test::RequestResultTable_Test()
    : Hanami::CompareTestHelper("RequestResultTable_Test")
{
    m_testUuid = generateUuid().toString();
    m_testName = "test-requestResult";

    addRequestResult_test();
    getRequestResult_test();
    getAllRequestResult_test();
    deleteRequestResult_test();
}

/**
 * @brief initTest
 */
void
RequestResultTable_Test::initTest()
{
    Hanami::ErrorContainer error;
    bool success = false;

    m_userContext.projectId = "test-project";
    m_userContext.userId = "test-user";

    if (Hanami::deleteFileOrDir(m_databasePath, error) == false) {
        LOG_DEBUG("No test-database to delete");
    }

    // initalize database
    m_database = Hanami::SqlDatabase::getInstance();
    if (m_database->initDatabase(m_databasePath, error) == false) {
        error.addMessage("Failed to initialize sql-database.");
        LOG_ERROR(error);
        assert(false);
    }
}

/**
 * @brief createTestDb
 */
void
RequestResultTable_Test::createTestDb()
{
    RequestResultTable::ResultDbEntry requestResultData;
    Hanami::ErrorContainer error;

    requestResultData.uuid = m_testUuid;
    requestResultData.name = m_testName;
    requestResultData.visibility = "private";
    requestResultData.data = "[1,1,1]";

    RequestResultTable* requestResultTable = RequestResultTable::getInstance();
    requestResultTable->initTable(error);
    requestResultTable->addRequestResult(requestResultData, m_userContext, error);

    requestResultData.uuid = generateUuid().toString();
    requestResultData.name = "test-requestResult2";
    requestResultTable->addRequestResult(requestResultData, m_userContext, error);
}

/**
 * @brief cleanupTest
 */
void
RequestResultTable_Test::cleanupTest()
{
    Hanami::ErrorContainer error;

    if (m_database->closeDatabase() == false) {
        error.addMessage("Failed to close test-database.");
        LOG_ERROR(error);
        assert(false);
    }

    if (Hanami::deleteFileOrDir(m_databasePath, error) == false) {
        error.addMessage("Failed to delete database-test-file.");
        LOG_ERROR(error);
        assert(false);
    }
}

/**
 * @brief addRequestResult_test
 */
void
RequestResultTable_Test::addRequestResult_test()
{
    initTest();

    RequestResultTable::ResultDbEntry requestResultData;
    Hanami::ErrorContainer error;

    requestResultData.uuid = generateUuid().toString();
    requestResultData.name = "test-requestResult";
    requestResultData.visibility = "private";
    requestResultData.data = "[1,1,1]";

    RequestResultTable* requestResultTable = RequestResultTable::getInstance();
    TEST_EQUAL(requestResultTable->initTable(error), true);
    TEST_EQUAL(requestResultTable->addRequestResult(requestResultData, m_userContext, error), OK);
    TEST_EQUAL(requestResultTable->addRequestResult(requestResultData, m_userContext, error),
               INVALID_INPUT);

    cleanupTest();
}

/**
 * @brief getRequestResult_test
 */
void
RequestResultTable_Test::getRequestResult_test()
{
    initTest();

    Hanami::ErrorContainer error;
    RequestResultTable* requestResultTable = RequestResultTable::getInstance();

    createTestDb();

    // positive test
    RequestResultTable::ResultDbEntry result;
    TEST_EQUAL(requestResultTable->getRequestResult(result, m_testUuid, m_userContext, error), OK);
    TEST_EQUAL(result.uuid, m_testUuid);
    TEST_EQUAL(result.name, m_testName);
    TEST_EQUAL(result.ownerId, m_userContext.userId);
    TEST_EQUAL(result.projectId, m_userContext.projectId);
    TEST_EQUAL(result.visibility, "private");
    TEST_EQUAL(result.data.dump(), "[1,1,1]");

    // negative test
    TEST_EQUAL(requestResultTable->getRequestResult(
                   result, generateUuid().toString(), m_userContext, error),
               INVALID_INPUT);

    cleanupTest();
}

/**
 * @brief getAllRequestResult_test
 */
void
RequestResultTable_Test::getAllRequestResult_test()
{
    initTest();

    Hanami::ErrorContainer error;
    RequestResultTable* requestResultTable = RequestResultTable::getInstance();

    createTestDb();

    Hanami::TableItem result;
    TEST_EQUAL(requestResultTable->getAllRequestResult(result, m_userContext, error), true);
    TEST_EQUAL(result.getNumberOfRows(), 2);
    TEST_EQUAL(result.getNumberOfColums(), requestResultTable->getNumberOfColumns() - 1);

    cleanupTest();
}

/**
 * @brief deleteRequestResult_test
 */
void
RequestResultTable_Test::deleteRequestResult_test()
{
    initTest();

    Hanami::ErrorContainer error;
    RequestResultTable* requestResultTable = RequestResultTable::getInstance();

    createTestDb();

    RequestResultTable::ResultDbEntry result;
    TEST_EQUAL(requestResultTable->deleteRequestResult(m_testUuid, m_userContext, error), OK);
    TEST_EQUAL(requestResultTable->deleteRequestResult(m_testUuid, m_userContext, error),
               INVALID_INPUT);
    TEST_EQUAL(requestResultTable->getRequestResult(result, m_testUuid, m_userContext, error),
               INVALID_INPUT);

    Hanami::TableItem result2;
    TEST_EQUAL(requestResultTable->getAllRequestResult(result2, m_userContext, error), true);
    TEST_EQUAL(result2.getNumberOfRows(), 1);
    TEST_EQUAL(result2.getNumberOfColums(), requestResultTable->getNumberOfColumns() - 1);

    cleanupTest();
}
