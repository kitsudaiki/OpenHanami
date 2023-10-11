/**
 * @file        task_handle_state.h
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

#ifndef HANAMI_TASKHANGLESTATE_H
#define HANAMI_TASKHANGLESTATE_H

#include <core/cluster/task.h>
#include <hanami_common/threading/event.h>

#include <algorithm>
#include <deque>
#include <map>
#include <mutex>

class Cluster;

class TaskHandle_State : public Hanami::Event
{
   public:
    TaskHandle_State(Cluster *cluster);
    ~TaskHandle_State();

    bool processEvent();

    bool addTask(const std::string &uuid, const Task &task);
    Task *getActualTask();

    const TaskProgress getProgress(const std::string &taskUuid);
    bool removeTask(const std::string &taskUuid);
    bool isFinish(const std::string &taskUuid);
    TaskState getTaskState(const std::string &taskUuid);
    void getAllProgress(std::map<std::string, TaskProgress> &result);

   private:
    Cluster *m_cluster = nullptr;

    std::deque<std::string> m_taskQueue;
    std::map<std::string, Task> m_taskMap;
    std::mutex m_task_mutex;
    Task *actualTask = nullptr;
    bool m_abort = false;

    bool getNextTask();
    void finishTask();
};

#endif  // HANAMI_TASKHANGLESTATE_H
