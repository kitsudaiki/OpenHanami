/**
 * @file        processing_unit_handler.cpp
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

#include <core/processing/processing_unit_handler.h>
#include <core/processing/cpu_processing_unit.h>

ProcessingUnitHandler* ProcessingUnitHandler::instance = nullptr;

/**
 * @brief constructor
 */
ProcessingUnitHandler::ProcessingUnitHandler() {}

/**
 * @brief destructor
 */
ProcessingUnitHandler::~ProcessingUnitHandler() {}

/**
 * @brief create new processing-thread and bind it to a cpu-thread
 *
 * @param threadId cpu-thread where the processing-thread should be connected to
 */
void
ProcessingUnitHandler::addProcessingUnit(const uint64_t threadId)
{
    CpuProcessingUnit* newUnit = new CpuProcessingUnit();
    m_processingUnits.push_back(newUnit);
    newUnit->startThread();
    newUnit->bindThreadToCore(threadId);
}

/**
 * @brief init processing-threads
 *
 * @param numberOfThreads number of threads to create
 *
 * @return always true
 */
bool
ProcessingUnitHandler::initProcessingUnits(const uint16_t numberOfThreads)
{
    // init cpu
    for(uint16_t i = 0; i < numberOfThreads; i++)
    {
        CpuProcessingUnit* newUnit = new CpuProcessingUnit();
        m_processingUnits.push_back(newUnit);
        newUnit->startThread();
    }

    return true;
}
