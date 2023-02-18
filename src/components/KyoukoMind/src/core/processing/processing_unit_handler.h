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

#ifndef KYOUKOMIND_PROCESSING_UNIT_HANDLER_H
#define KYOUKOMIND_PROCESSING_UNIT_HANDLER_H

#include <vector>
#include <stdint.h>

class CpuProcessingUnit;

class ProcessingUnitHandler
{
public:
    ProcessingUnitHandler();
    ~ProcessingUnitHandler();

    bool initProcessingUnits(const uint16_t numberOfThreads);

private:
    std::vector<CpuProcessingUnit*> m_processingUnits;
};

#endif // KYOUKOMIND_PROCESSING_UNIT_HANDLER_H
