/**
 * @file        checkpoint_table_test.cpp
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

#include "checkpoint_table_test.h"

#include <hanami_common/functions/file_functions.h>
#include <hanami_common/logger.h>
#include <hanami_common/uuid.h>
#include <hanami_database/sql_database.h>
#include <src/database/checkpoint_table.h>

CheckpointTable_Test::CheckpointTable_Test() : Hanami::CompareTestHelper("CheckpointTable_Test")
{
    m_testUuid = generateUuid().toString();
    m_testName = "test-checkpoint";

    addCheckpoint_test();
    getCheckpoint_test();
    getAllCheckpoint_test();
    deleteCheckpoint_test();
}

/**
 * @brief initTest
 */
void
CheckpointTable_Test::initTest()
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
CheckpointTable_Test::createTestDb()
{
    CheckpointTable::CheckpointDbEntry checkpointData;
    Hanami::ErrorContainer error;

    checkpointData.uuid = m_testUuid;
    checkpointData.name = m_testName;
    checkpointData.visibility = "private";
    checkpointData.location = "/tmp/test-checkpoint";

    CheckpointTable* checkpointTable = CheckpointTable::getInstance();
    checkpointTable->initTable(error);
    checkpointTable->addCheckpoint(checkpointData, m_userContext, error);

    checkpointData.uuid = generateUuid().toString();
    checkpointData.name = "test-checkpoint2";
    checkpointData.location = "/tmp/test-checkpoint2";
    checkpointTable->addCheckpoint(checkpointData, m_userContext, error);
}

/**
 * @brief cleanupTest
 */
void
CheckpointTable_Test::cleanupTest()
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
 * @brief addCheckpoint_test
 */
void
CheckpointTable_Test::addCheckpoint_test()
{
    initTest();

    CheckpointTable::CheckpointDbEntry checkpointData;
    Hanami::ErrorContainer error;

    checkpointData.uuid = generateUuid().toString();
    checkpointData.name = "test-checkpoint";
    checkpointData.visibility = "private";
    checkpointData.location = "/tmp/test-checkpoint";

    CheckpointTable* checkpointTable = CheckpointTable::getInstance();
    TEST_EQUAL(checkpointTable->initTable(error), true);
    TEST_EQUAL(checkpointTable->addCheckpoint(checkpointData, m_userContext, error), OK);
    TEST_EQUAL(checkpointTable->addCheckpoint(checkpointData, m_userContext, error), INVALID_INPUT);

    cleanupTest();
}

/**
 * @brief getCheckpoint_test
 */
void
CheckpointTable_Test::getCheckpoint_test()
{
    initTest();

    Hanami::ErrorContainer error;
    CheckpointTable* checkpointTable = CheckpointTable::getInstance();

    createTestDb();

    // positive test
    CheckpointTable::CheckpointDbEntry result;
    TEST_EQUAL(checkpointTable->getCheckpoint(result, m_testUuid, m_userContext, error), OK);
    TEST_EQUAL(result.uuid, m_testUuid);
    TEST_EQUAL(result.name, m_testName);
    TEST_EQUAL(result.visibility, "private");
    TEST_EQUAL(result.ownerId, m_userContext.userId);
    TEST_EQUAL(result.projectId, m_userContext.projectId);
    TEST_EQUAL(result.location, "/tmp/test-checkpoint");

    // negative test
    TEST_EQUAL(
        checkpointTable->getCheckpoint(result, generateUuid().toString(), m_userContext, error),
        INVALID_INPUT);

    cleanupTest();
}

/**
 * @brief getAllCheckpoint_test
 */
void
CheckpointTable_Test::getAllCheckpoint_test()
{
    initTest();

    Hanami::ErrorContainer error;
    CheckpointTable* checkpointTable = CheckpointTable::getInstance();

    createTestDb();

    Hanami::TableItem result;
    TEST_EQUAL(checkpointTable->getAllCheckpoint(result, m_userContext, true, error), true);
    TEST_EQUAL(result.getNumberOfRows(), 2);
    TEST_EQUAL(result.getNumberOfColums(), checkpointTable->getNumberOfColumns());

    result.clearTable();
    TEST_EQUAL(checkpointTable->getAllCheckpoint(result, m_userContext, false, error), true);
    TEST_EQUAL(result.getNumberOfRows(), 2);
    TEST_EQUAL(result.getNumberOfColums(), checkpointTable->getNumberOfColumns() - 1);

    cleanupTest();
}

/**
 * @brief deleteCheckpoint_test
 */
void
CheckpointTable_Test::deleteCheckpoint_test()
{
    initTest();

    Hanami::ErrorContainer error;
    CheckpointTable* checkpointTable = CheckpointTable::getInstance();

    createTestDb();

    CheckpointTable::CheckpointDbEntry result;
    TEST_EQUAL(checkpointTable->deleteCheckpoint(m_testUuid, m_userContext, error), OK);
    TEST_EQUAL(checkpointTable->deleteCheckpoint(m_testUuid, m_userContext, error), OK);
    TEST_EQUAL(checkpointTable->getCheckpoint(result, m_testUuid, m_userContext, error),
               INVALID_INPUT);

    Hanami::TableItem result2;
    TEST_EQUAL(checkpointTable->getAllCheckpoint(result2, m_userContext, true, error), true);
    TEST_EQUAL(result2.getNumberOfRows(), 1);
    TEST_EQUAL(result2.getNumberOfColums(), checkpointTable->getNumberOfColumns());

    cleanupTest();
}
