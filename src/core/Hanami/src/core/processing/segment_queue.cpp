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

#include <core/segments/core_segment/core_segment.h>

SegmentQueue* SegmentQueue::instance = nullptr;

/**
 * @brief constructor
 */
SegmentQueue::SegmentQueue() {}

/**
 * @brief add segment to queue
 *
 * @param newSegment segment to add to queue
 */
void
SegmentQueue::addSegmentToQueue(CoreSegment* newSegment)
{
    while(m_queue_lock.test_and_set(std::memory_order_acquire)) { asm(""); }

    m_segmentQueue.push_back(newSegment);
    m_queue_lock.clear(std::memory_order_release);
}

/**
 * @brief add a list of segments to the queue
 *
 * @param semgnetList list with segments to add
 */
void
SegmentQueue::addSegmentListToQueue(const std::vector<CoreSegment*> &semgnetList)
{
    while(m_queue_lock.test_and_set(std::memory_order_acquire)) { asm(""); }

    for(CoreSegment* segment : semgnetList) {
        m_segmentQueue.push_back(segment);
    }

    m_queue_lock.clear(std::memory_order_release);
}

/**
 * @brief get next segment in the queue
 *
 * @return nullptr, if queue is empty, else next segment in queue
 */
CoreSegment*
SegmentQueue::getSegmentFromQueue()
{
    CoreSegment* result = nullptr;

    while(m_queue_lock.test_and_set(std::memory_order_acquire)) { asm(""); }

    if(m_segmentQueue.size() > 0)
    {
        result = m_segmentQueue.front();
        m_segmentQueue.pop_front();
    }

    m_queue_lock.clear(std::memory_order_release);

    return result;
}
