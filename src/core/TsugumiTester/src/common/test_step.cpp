/**
 * @file        test_step.cpp
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

#include "test_step.h"

/**
 * @brief constructor
 *
 * @param expectSuccess sould success or fail
 */
TestStep::TestStep(const bool expectSuccess)
    : m_expectSuccess(expectSuccess) {}

/**
 * @brief destructor
 */
TestStep::~TestStep() {}

/**
 * @brief get test-name
 *
 * @return name of the test
 */
const std::string
TestStep::getTestName() const
{
    return m_testName;
}
