/**
 * @file        objects.h
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

#ifndef HANAMI_OUTPUT_SEGMENT_OBJECTS_H
#define HANAMI_OUTPUT_SEGMENT_OBJECTS_H

#include <common.h>

struct OutputNeuron
{
    float outputWeight = 0.0f;
    float shouldValue = 0.0f;
    uint32_t targetBorderId = 0;    
    float maxWeight = 0.00001f;

    // total size: 16 Byte
};

//==================================================================================================

#endif // HANAMI_OUTPUT_SEGMENT_OBJECTS_H
