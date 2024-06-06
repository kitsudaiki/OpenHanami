/**
 * @file        dataset_io_test.cpp
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

#include "dataset_io_test.h"

#include <core/io/data_set/dataset_file_io.h>
#include <hanami_common/functions/file_functions.h>

DataSetIO_Test::DataSetIO_Test() : Hanami::CompareTestHelper("DataSetIO_Test")
{
    m_input = json::array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    assert(m_input.size() > 0 && m_input.size() % 3 == 0);

    write_test();
    read_test();
}

/**
 * @brief write_test
 */
void
DataSetIO_Test::write_test()
{
    Hanami::ErrorContainer error;
    DataSetFileHandle fileHandle;

    Hanami::deleteFileOrDir(m_testFilePath, error);

    TEST_EQUAL(initNewDataSetFile(
                   fileHandle, m_testFilePath, m_fileName, FLOAT_TYPE, m_numberOfColumns, error),
               OK);
    TEST_EQUAL(initNewDataSetFile(
                   fileHandle, m_testFilePath, m_fileName, FLOAT_TYPE, m_numberOfColumns, error),
               INVALID_INPUT);

    TEST_EQUAL(fileHandle.header.getName(), m_fileName);
    TEST_EQUAL(fileHandle.header.fileSize, sizeof(DataSetHeader));
    TEST_EQUAL(fileHandle.header.dataType, FLOAT_TYPE);
    TEST_EQUAL(fileHandle.header.numberOfColumns, m_numberOfColumns);
    TEST_EQUAL(fileHandle.header.numberOfRows, 0);

    TEST_EQUAL(appendToDataSet<float>(fileHandle, m_input, error), OK);

    TEST_EQUAL(fileHandle.header.getName(), m_fileName);
    TEST_EQUAL(fileHandle.header.fileSize,
               sizeof(DataSetHeader) + (m_input.size() * sizeof(float)));
    TEST_EQUAL(fileHandle.header.dataType, FLOAT_TYPE);
    TEST_EQUAL(fileHandle.header.numberOfColumns, m_numberOfColumns);
    TEST_EQUAL(fileHandle.header.numberOfRows, 3);
}

/**
 * @brief read_test
 */
void
DataSetIO_Test::read_test()
{
    Hanami::ErrorContainer error;
    DataSetFileHandle fileHandle;

    TEST_EQUAL(openDataSetFile(fileHandle, m_testFilePath, error), OK);

    TEST_EQUAL(fileHandle.header.getName(), m_fileName);
    TEST_EQUAL(fileHandle.header.fileSize,
               sizeof(DataSetHeader) + (m_input.size() * sizeof(float)));
    TEST_EQUAL(fileHandle.header.dataType, FLOAT_TYPE);
    TEST_EQUAL(fileHandle.header.numberOfColumns, m_numberOfColumns);
    TEST_EQUAL(fileHandle.header.numberOfRows, 3);

    std::vector<float> output;
    DataSetSelector selector;
    selector.endColumn = 3;
    selector.endRow = 3;
    TEST_EQUAL(getDataFromDataSet(output, fileHandle, selector, error), OK);
    TEST_EQUAL(output.size(), m_input.size());

    for (uint64_t i = 0; i < output.size(); i++) {
        TEST_EQUAL(output[i], static_cast<float>(m_input[i]));
    }

    selector.endRow = 2;
    TEST_EQUAL(getDataFromDataSet(output, fileHandle, selector, error), OK);
    TEST_EQUAL(output.size(), m_input.size() - 3);

    for (uint64_t i = 0; i < output.size() - 3; i++) {
        TEST_EQUAL(output[i], static_cast<float>(m_input[i]));
    }

    selector.startRow = 1;
    selector.endRow = 2;
    selector.startColumn = 1;
    selector.endColumn = 2;
    TEST_EQUAL(getDataFromDataSet(output, fileHandle, selector, error), OK);
    TEST_EQUAL(output.size(), 1);
    TEST_EQUAL(output[0], static_cast<float>(m_input[4]));
}
