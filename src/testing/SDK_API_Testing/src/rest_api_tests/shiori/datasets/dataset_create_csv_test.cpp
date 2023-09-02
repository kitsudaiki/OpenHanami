/**
 * @file        dataset_create_csv_test.cpp
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

#include "dataset_create_csv_test.h"

#include <hanami_config/config_handler.h>
#include <hanami_sdk/data_set.h>

DataSetCreateCsvTest::DataSetCreateCsvTest(const bool expectSuccess)
          : TestStep(expectSuccess)
{
    m_testName = "create csv data-set";
    if(expectSuccess) {
        m_testName += " (success)";
    } else {
        m_testName += " (fail)";
    }
}

bool
DataSetCreateCsvTest::runTest(Hanami::JsonItem &inputData,
                              Hanami::ErrorContainer &error)
{
    std::string result;
    if(HanamiAI::uploadCsvData(result,
                               inputData.get("base_dataset_name").getString(),
                               inputData.get("base_inputs").getString(),
                               error) != m_expectSuccess)
    {
        return false;
    }

    if(m_expectSuccess == false) {
        return true;
    }

    // parse output
    Hanami::JsonItem jsonItem;
    if(jsonItem.parse(result, error) == false) {
        return false;
    }

    inputData.insert("base_dataset_uuid", jsonItem.get("uuid").getString(), true);

    return true;
}
