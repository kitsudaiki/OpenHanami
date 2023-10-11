/**
 * @file        test_thread.h
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

#ifndef TSUGUMI_TESTTHREAD_H
#define TSUGUMI_TESTTHREAD_H

#include <hanami_common/threading/thread.h>
#include <hanami_sdk/common/websocket_client.h>

#include <deque>
#include <mutex>

class TestStep;

class TestThread : public Hanami::Thread
{
   public:
    TestThread(const std::string& name, json& inputData);
    ~TestThread();

    void addTest(TestStep* newStep);

    bool isFinished = false;
    static Hanami::WebsocketClient* m_wsClient;

   protected:
    void run();

   private:
    std::deque<TestStep*> m_taskQueue;
    std::mutex m_queueLock;
    json m_inputData;

    TestStep* getTest();
};

#endif  // TSUGUMI_TESTTHREAD_H
