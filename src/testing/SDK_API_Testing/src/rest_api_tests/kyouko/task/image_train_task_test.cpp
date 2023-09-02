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

#include <libHanamiAiSdk/task.h>

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
ImageTrainTaskTest::runTest(Hanami::JsonItem &inputData,
                            Hanami::ErrorContainer &error)
{
    // create new user
    std::string result;
    if(HanamiAI::createTask(result,
                            inputData.get("generic_task_name").getString(),
                            "train",
                            inputData.get("cluster_uuid").getString(),
                            inputData.get("train_dataset_uuid").getString(),
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

    inputData.insert("train_task_uuid", jsonItem.get("uuid").getString(), true);

    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    start = std::chrono::system_clock::now();

    // wait until task is finished
    do
    {
        usleep(1000000);

        HanamiAI::getTask(result,
                          inputData.get("train_task_uuid").getString(),
                          inputData.get("cluster_uuid").getString(),
                          error);

        // parse output
        if(jsonItem.parse(result, error) == false) {
            return false;
        }
        //std::cout<<jsonItem.toString(true)<<std::endl;
    }
    while(jsonItem.get("state").getString() != "finished");
    end = std::chrono::system_clock::now();
    const float time2 = std::chrono::duration_cast<chronoMilliSec>(end - start).count();

    std::cout<<"#######################################################################"<<std::endl;
    std::cout<<"train_time: "<<time2<<" ms"<<std::endl;
    std::cout<<"#######################################################################"<<std::endl;

    return true;
}
