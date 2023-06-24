/**
 * @file        processing_unit_handler.h
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

#ifndef HANAMI_SEGMENTQUEUE_H
#define HANAMI_SEGMENTQUEUE_H

#include <vector>
#include <deque>
#include <atomic>

class AbstractSegment;

class SegmentQueue
{
public:
    SegmentQueue();

    void addSegmentToQueue(AbstractSegment* newSegment);
    void addSegmentListToQueue(const std::vector<AbstractSegment*> &semgnetList);

    AbstractSegment* getSegmentFromQueue();

private:
    std::atomic_flag m_queue_lock = ATOMIC_FLAG_INIT;
    std::deque<AbstractSegment*> m_segmentQueue;
};

#endif // HANAMI_SEGMENTQUEUE_H
