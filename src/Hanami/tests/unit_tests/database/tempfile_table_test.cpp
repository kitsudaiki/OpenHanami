/**
 * @file        tempfile_table_test.cpp
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

#include "tempfile_table_test.h"

#include <hanami_common/functions/file_functions.h>
#include <hanami_common/logger.h>
#include <hanami_common/uuid.h>
#include <hanami_database/sql_database.h>
#include <src/database/tempfile_table.h>

TempfileTable_Test::TempfileTable_Test() : Hanami::CompareTestHelper("TempfileTable_Test")
{
    m_testUuid1 = generateUuid().toString();
    m_testUuid2 = generateUuid().toString();
    m_testName = "test-tempfile";
    m_testResourceUuid = generateUuid().toString();

    addTempfile_test();
    getTempfile_test();
    getAllTempfile_test();
    deleteTempfile_test();
    getRelatedResourceUuids_test();
}

/**
 * @brief initTest
 */
void
TempfileTable_Test::initTest()
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
TempfileTable_Test::createTestDb()
{
    TempfileTable::TempfileDbEntry tempfileData;
    Hanami::ErrorContainer error;

    tempfileData.uuid = m_testUuid1;
    tempfileData.name = m_testName;
    tempfileData.visibility = "private";
    tempfileData.relatedResourceType = "dataset";
    tempfileData.relatedResourceUuid = m_testResourceUuid;
    tempfileData.location = "/tmp/tempfile";
    tempfileData.fileSize = 42;

    TempfileTable* tempfileTable = TempfileTable::getInstance();
    tempfileTable->initTable(error);
    tempfileTable->addTempfile(tempfileData, m_userContext, error);

    tempfileData.uuid = m_testUuid2;
    tempfileData.name = "test-tempfile2";
    tempfileTable->addTempfile(tempfileData, m_userContext, error);
}

/**
 * @brief cleanupTest
 */
void
TempfileTable_Test::cleanupTest()
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
 * @brief addTempfile_test
 */
void
TempfileTable_Test::addTempfile_test()
{
    initTest();

    TempfileTable::TempfileDbEntry tempfileData;
    Hanami::ErrorContainer error;

    tempfileData.uuid = generateUuid().toString();
    tempfileData.name = "test-tempfile";
    tempfileData.visibility = "private";
    tempfileData.relatedResourceType = "dataset";
    tempfileData.relatedResourceUuid = m_testResourceUuid;
    tempfileData.location = "/tmp/tempfile";
    tempfileData.fileSize = 42;

    TempfileTable* tempfileTable = TempfileTable::getInstance();
    TEST_EQUAL(tempfileTable->initTable(error), true);
    TEST_EQUAL(tempfileTable->addTempfile(tempfileData, m_userContext, error), OK);
    TEST_EQUAL(tempfileTable->addTempfile(tempfileData, m_userContext, error), INVALID_INPUT);

    cleanupTest();
}

/**
 * @brief getTempfile_test
 */
void
TempfileTable_Test::getTempfile_test()
{
    initTest();

    Hanami::ErrorContainer error;
    TempfileTable* tempfileTable = TempfileTable::getInstance();

    createTestDb();

    // positive test
    TempfileTable::TempfileDbEntry result;
    TEST_EQUAL(tempfileTable->getTempfile(result, m_testUuid1, m_userContext, error), OK);
    TEST_EQUAL(result.uuid, m_testUuid1);
    TEST_EQUAL(result.name, m_testName);
    TEST_EQUAL(result.visibility, "private");
    TEST_EQUAL(result.relatedResourceType, "dataset");
    TEST_EQUAL(result.relatedResourceUuid, m_testResourceUuid);
    TEST_EQUAL(result.location, "/tmp/tempfile");
    TEST_EQUAL(result.fileSize, 42);

    // negative test
    TEST_EQUAL(tempfileTable->getTempfile(result, generateUuid().toString(), m_userContext, error),
               INVALID_INPUT);

    cleanupTest();
}

/**
 * @brief getAllTempfile_test
 */
void
TempfileTable_Test::getAllTempfile_test()
{
    initTest();

    Hanami::ErrorContainer error;
    TempfileTable* tempfileTable = TempfileTable::getInstance();

    createTestDb();

    Hanami::TableItem result;
    TEST_EQUAL(tempfileTable->getAllTempfile(result, m_userContext, error), true);
    TEST_EQUAL(result.getNumberOfRows(), 2);
    TEST_EQUAL(result.getNumberOfColums(), tempfileTable->getNumberOfColumns() - 1);

    cleanupTest();
}

/**
 * @brief deleteTempfile_test
 */
void
TempfileTable_Test::deleteTempfile_test()
{
    initTest();

    Hanami::ErrorContainer error;
    TempfileTable* tempfileTable = TempfileTable::getInstance();

    createTestDb();

    TempfileTable::TempfileDbEntry result;
    TEST_EQUAL(tempfileTable->deleteTempfile(m_testUuid1, m_userContext, error), OK);
    TEST_EQUAL(tempfileTable->deleteTempfile(m_testUuid1, m_userContext, error), OK);
    TEST_EQUAL(tempfileTable->getTempfile(result, m_testUuid1, m_userContext, error),
               INVALID_INPUT);

    Hanami::TableItem result2;
    TEST_EQUAL(tempfileTable->getAllTempfile(result2, m_userContext, error), true);
    TEST_EQUAL(result2.getNumberOfRows(), 1);
    TEST_EQUAL(result2.getNumberOfColums(), tempfileTable->getNumberOfColumns() - 1);

    cleanupTest();
}

/**
 * @brief getRelatedResourceUuids_test
 */
void
TempfileTable_Test::getRelatedResourceUuids_test()
{
    initTest();

    Hanami::ErrorContainer error;
    TempfileTable* tempfileTable = TempfileTable::getInstance();

    createTestDb();

    std::vector<std::string> relatedUuids;
    TEST_EQUAL(tempfileTable->getRelatedResourceUuids(
                   relatedUuids, "dataset", m_testResourceUuid, m_userContext, error),
               OK);
    TEST_EQUAL(relatedUuids.size(), 2);
    TEST_EQUAL(relatedUuids.at(0), m_testUuid1);
    TEST_EQUAL(relatedUuids.at(1), m_testUuid2);

    cleanupTest();
}
