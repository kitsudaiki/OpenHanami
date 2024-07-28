/**
 * @file        dataset_table_test.cpp
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

#include "dataset_table_test.h"

#include <hanami_common/functions/file_functions.h>
#include <hanami_common/logger.h>
#include <hanami_common/uuid.h>
#include <hanami_database/sql_database.h>
#include <src/database/dataset_table.h>

DataSetTable_Test::DataSetTable_Test() : Hanami::CompareTestHelper("DataSetTable_Test")
{
    m_testUuid = generateUuid().toString();
    m_testName = "test-dataset";

    addDataSet_test();
    getDataSet_test();
    getAllDataSet_test();
    deleteDataSet_test();
}

/**
 * @brief initTest
 */
void
DataSetTable_Test::initTest()
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
DataSetTable_Test::createTestDb()
{
    DataSetTable::DataSetDbEntry datasetData;
    Hanami::ErrorContainer error;

    datasetData.uuid = m_testUuid;
    datasetData.name = m_testName;
    datasetData.visibility = "private";
    datasetData.location = "/etc/temp";

    DataSetTable* datasetTable = DataSetTable::getInstance();
    datasetTable->initTable(error);
    datasetTable->addDataSet(datasetData, m_userContext, error);

    datasetData.uuid = generateUuid().toString();
    datasetData.name = "test-dataset2";
    datasetTable->addDataSet(datasetData, m_userContext, error);
}

/**
 * @brief cleanupTest
 */
void
DataSetTable_Test::cleanupTest()
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
 * @brief addDataSet_test
 */
void
DataSetTable_Test::addDataSet_test()
{
    initTest();

    DataSetTable::DataSetDbEntry datasetData;
    Hanami::ErrorContainer error;

    datasetData.uuid = generateUuid().toString();
    datasetData.name = "test-dataset";
    datasetData.visibility = "private";
    datasetData.location = "/etc/temp";

    DataSetTable* datasetTable = DataSetTable::getInstance();
    TEST_EQUAL(datasetTable->initTable(error), true);
    TEST_EQUAL(datasetTable->addDataSet(datasetData, m_userContext, error), OK);
    TEST_EQUAL(datasetTable->addDataSet(datasetData, m_userContext, error), INVALID_INPUT);

    cleanupTest();
}

/**
 * @brief getDataSet_test
 */
void
DataSetTable_Test::getDataSet_test()
{
    initTest();

    Hanami::ErrorContainer error;
    DataSetTable* datasetTable = DataSetTable::getInstance();

    createTestDb();

    // positive test
    DataSetTable::DataSetDbEntry result;
    TEST_EQUAL(datasetTable->getDataSet(result, m_testUuid, m_userContext, error), OK);
    TEST_EQUAL(result.uuid, m_testUuid);
    TEST_EQUAL(result.name, m_testName);
    TEST_EQUAL(result.visibility, "private");
    TEST_EQUAL(result.ownerId, m_userContext.userId);
    TEST_EQUAL(result.projectId, m_userContext.projectId);
    TEST_EQUAL(result.location, "/etc/temp");

    // negative test
    TEST_EQUAL(datasetTable->getDataSet(result, generateUuid().toString(), m_userContext, error),
               INVALID_INPUT);

    cleanupTest();
}

/**
 * @brief getAllDataSet_test
 */
void
DataSetTable_Test::getAllDataSet_test()
{
    initTest();

    Hanami::ErrorContainer error;
    DataSetTable* datasetTable = DataSetTable::getInstance();

    createTestDb();

    Hanami::TableItem result;
    TEST_EQUAL(datasetTable->getAllDataSet(result, m_userContext, error), true);
    TEST_EQUAL(result.getNumberOfRows(), 2);
    TEST_EQUAL(result.getNumberOfColums(), datasetTable->getNumberOfColumns() - 1);

    cleanupTest();
}

/**
 * @brief deleteDataSet_test
 */
void
DataSetTable_Test::deleteDataSet_test()
{
    initTest();

    Hanami::ErrorContainer error;
    DataSetTable* datasetTable = DataSetTable::getInstance();

    createTestDb();

    DataSetTable::DataSetDbEntry result;
    TEST_EQUAL(datasetTable->deleteDataSet(m_testUuid, m_userContext, error), OK);
    TEST_EQUAL(datasetTable->deleteDataSet(m_testUuid, m_userContext, error), INVALID_INPUT);
    TEST_EQUAL(datasetTable->getDataSet(result, m_testUuid, m_userContext, error), INVALID_INPUT);

    Hanami::TableItem result2;
    TEST_EQUAL(datasetTable->getAllDataSet(result2, m_userContext, error), true);
    TEST_EQUAL(result2.getNumberOfRows(), 1);
    TEST_EQUAL(result2.getNumberOfColums(), datasetTable->getNumberOfColumns() - 1);

    cleanupTest();
}
