/**
 * @file        table_train_task_test.h
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

#ifndef TSUGUMITESTER_TABLE_TRAINTASKTEST_H
#define TSUGUMITESTER_TABLE_TRAINTASKTEST_H

#include <common/test_step.h>

class TableTrainTaskTest : public TestStep
{
   public:
    TableTrainTaskTest(const bool expectSuccess);

    bool runTest(json& inputData, Hanami::ErrorContainer& error);
};

#endif  // TSUGUMITESTER_TABLE_TRAINTASKTEST_H
