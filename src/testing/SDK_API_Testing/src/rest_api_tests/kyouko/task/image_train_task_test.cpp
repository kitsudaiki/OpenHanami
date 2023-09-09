/**
 * @file        train_task_test.cpp
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

#include "image_train_task_test.h"
#include <chrono>

#include <hanami_sdk/task.h>

typedef std::chrono::milliseconds chronoMilliSec;
typedef std::chrono::microseconds chronoMicroSec;
typedef std::chrono::nanoseconds chronoNanoSec;
typedef std::chrono::seconds chronoSec;
typedef std::chrono::high_resolution_clock::time_point chronoTimePoint;
typedef std::chrono::high_resolution_clock chronoClock;


ImageTrainTaskTest::ImageTrainTaskTest(const bool expectSuccess)
  : TestStep(expectSuccess)
{
    m_testName = "train-task";
    if(expectSuccess) {
        m_testName += " (success)";
    } else {
        m_testName += " (fail)";
    }
}

bool
ImageTrainTaskTest::runTest(json &inputData,
                            Hanami::ErrorContainer &error)
{
    // create new user
    std::string result;
    if(Hanami::createTask(result,
                          inputData["generic_task_name"],
                          "train",
                          inputData["cluster_uuid"],
                          inputData["train_dataset_uuid"],
                          error) != m_expectSuccess)
    {
        return false;
    }

    if(m_expectSuccess == false) {
        return true;
    }

    // parse output
    json jsonItem = json::parse(result, nullptr, false);
    if (jsonItem.is_discarded())
    {
        std::cerr << "parse error" << std::endl;
        return false;
    }

    inputData["train_task_uuid"] = jsonItem["uuid"];

    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    start = std::chrono::system_clock::now();

    // wait until task is finished
    do
    {
        usleep(1000000);

        Hanami::getTask(result,
                        inputData["train_task_uuid"],
                        inputData["cluster_uuid"],
                        error);

        // parse output
        jsonItem = json::parse(result, nullptr, false);
        if (jsonItem.is_discarded())
        {
            std::cerr << "parse error" << std::endl;
            return false;
        }
        //std::cout<<jsonItem.dump(4)<<std::endl;
    }
    while(jsonItem["state"] != "finished");
    end = std::chrono::system_clock::now();
    const float time2 = std::chrono::duration_cast<chronoMilliSec>(end - start).count();

    std::cout<<"#######################################################################"<<std::endl;
    std::cout<<"train_time: "<<time2<<" ms"<<std::endl;
    std::cout<<"#######################################################################"<<std::endl;

    return true;
}
