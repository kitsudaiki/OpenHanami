/**
 * @file        cpu_processing_unit.h
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

#ifndef HANAMI_CPU_PROCESSING_UNIT_H
#define HANAMI_CPU_PROCESSING_UNIT_H

#include <common.h>
#include <hanami_common/threading/thread.h>

class Cluster;

class CpuProcessingUnit : public Hanami::Thread
{
   public:
    CpuProcessingUnit();
    ~CpuProcessingUnit();

   protected:
    void run();

   private:
    uint64_t reductionCounter = 0;

    void trainSegmentForward(Cluster* cluster);
    void trainSegmentBackward(Cluster* cluster);
    void processSegment(Cluster* cluster);
};

#endif  // HANAMI_CPU_PROCESSING_UNIT_H
