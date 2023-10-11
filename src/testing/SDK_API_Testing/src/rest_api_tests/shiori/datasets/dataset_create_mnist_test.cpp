/**
 * @file        dataset_create_mnist_test.cpp
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

#include "dataset_create_mnist_test.h"

#include <hanami_config/config_handler.h>
#include <hanami_sdk/data_set.h>

DataSetCreateMnistTest::DataSetCreateMnistTest(const bool expectSuccess, const std::string &type)
    : TestStep(expectSuccess)
{
    m_testName = "create mnist data-set";
    if (expectSuccess) {
        m_testName += " (success)";
    } else {
        m_testName += " (fail)";
    }
    m_type = type;
}

bool
DataSetCreateMnistTest::runTest(json &inputData, Hanami::ErrorContainer &error)
{
    std::string result;
    if (m_type == "train") {
        if (Hanami::uploadMnistData(result,
                                    inputData["train_dataset_name"],
                                    inputData["train_inputs"],
                                    inputData["train_labels"],
                                    error)
            != m_expectSuccess) {
            return false;
        }
    } else {
        if (Hanami::uploadMnistData(result,
                                    inputData["request_dataset_name"],
                                    inputData["request_inputs"],
                                    inputData["request_labels"],
                                    error)
            != m_expectSuccess) {
            return false;
        }
    }

    if (m_expectSuccess == false) {
        return true;
    }

    // parse output
    json jsonItem;
    try {
        jsonItem = json::parse(result);
    } catch (const json::parse_error &ex) {
        error.addMeesage("json-parser error: " + std::string(ex.what()));
        return false;
    }

    if (m_type == "train") {
        inputData["train_dataset_uuid"] = jsonItem["uuid"];
    } else {
        inputData["request_dataset_uuid"] = jsonItem["uuid"];
    }

    return true;
}
