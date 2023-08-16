/**
 * @file        segment_queue.h
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

#include "segment_queue.h"

#include <core/cluster/cluster.h>

ClusterQueue* ClusterQueue::instance = nullptr;

/**
 * @brief constructor
 */
ClusterQueue::ClusterQueue() {}

/**
 * @brief add segment to queue
 *
 * @param newSegment segment to add to queue
 */
void
ClusterQueue::addClusterToQueue(Cluster* newSegment)
{
    while(m_queue_lock.test_and_set(std::memory_order_acquire)) { asm(""); }

    m_clusterQueue.push_back(newSegment);
    m_queue_lock.clear(std::memory_order_release);
}

/**
 * @brief get next segment in the queue
 *
 * @return nullptr, if queue is empty, else next segment in queue
 */
Cluster*
ClusterQueue::getClusterFromQueue()
{
    Cluster* result = nullptr;

    while(m_queue_lock.test_and_set(std::memory_order_acquire)) { asm(""); }

    if(m_clusterQueue.size() > 0)
    {
        result = m_clusterQueue.front();
        m_clusterQueue.pop_front();
    }

    m_queue_lock.clear(std::memory_order_release);

    return result;
}
