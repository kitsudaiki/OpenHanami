/**
 * @file        test_thread.cpp
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

#include "test_thread.h"

#include <common/test_step.h>

Hanami::WebsocketClient* TestThread::m_wsClient = nullptr;

/**
 * @brief constructor
 */
TestThread::TestThread(const std::string &name,
                       json &inputData)
    : Hanami::Thread(name)
{
    m_inputData = inputData;
}

/**
 * @brief destructor
 */
TestThread::~TestThread()
{
    if(m_wsClient != nullptr) {
        delete m_wsClient;
    }
}

/**
 * @brief add new test to queue
 *
 * @param newStep new test-step to process
 */
void
TestThread::addTest(TestStep* newStep)
{
    std::lock_guard<std::mutex> guard(m_queueLock);

    m_taskQueue.push_back(newStep);
}

/**
 * @brief execute tests
 */
void
TestThread::run()
{
    TestStep* currentStep = getTest();
    while(currentStep != nullptr)
    {
        std::cout<<"==================================================================="<<std::endl;
        std::cout<<"run test: '"<<currentStep->getTestName()<<"'"<<std::endl;
        std::cout<<"-------------------------------------------------------------------"<<std::endl;

        // run test
        Hanami::ErrorContainer error;
        if(currentStep->runTest(m_inputData, error) == false)
        {
            error.addMeesage("Test '"
                             + currentStep->getTestName()
                             + "' in Thread '"
                             + getThreadName()
                             + "' has failed");
            LOG_ERROR(error);
            delete currentStep;
            std::cout<<std::endl;
            std::cout<<"RESULT: ERROR"<<std::endl;
            break;
        }
        else
        {
            std::cout<<std::endl;
            std::cout<<"RESULT: SUCCESS"<<std::endl;
        }

        std::cout<<std::endl;

        // get next test from queue
        delete currentStep;
        currentStep = getTest();
    }
    std::cout<<"==================================================================="<<std::endl;

    isFinished = true;
}

/**
 * @brief get next test from queue
 *
 * @return next test of queue
 */
TestStep*
TestThread::getTest()
{
    std::lock_guard<std::mutex> guard(m_queueLock);

    if(m_taskQueue.size() > 0)
    {
        TestStep* result = m_taskQueue.front();
        m_taskQueue.pop_front();
        return result;
    }

    return nullptr;
}
