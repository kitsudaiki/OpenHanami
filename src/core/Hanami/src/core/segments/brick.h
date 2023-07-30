/**
 * @file        brick.h
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

#ifndef HANAMI_BRICK_H
#define HANAMI_BRICK_H

#include "core_segment/objects.h"

#include <libKitsunemimiCommon/structs.h>

struct Brick
{
    // common
    uint32_t brickId = UNINIT_STATE_32;
    bool isOutputBrick = false;
    bool isInputBrick = false;
    uint8_t padding1[10];
    uint32_t numberOfPossibleBricks = 0;
    uint32_t neuronSectionPos = UNINIT_STATE_32;

    uint32_t numberOfNeurons = 0;
    uint32_t numberOfNeuronSections = 0;

    Kitsunemimi::Position brickPos;
    uint32_t neighbors[12];
    uint32_t possibleTargetBrickIds[1000];

    Brick()
    {
        for(uint32_t i = 0; i < 1000; i++) {
            possibleTargetBrickIds[i] = UNINIT_STATE_32;
        }
    }

    void addPossibleBrick(const uint32_t possibleBrickId)
    {
        for(uint32_t i = 0; i < 1000-1; i++)
        {
            if(possibleTargetBrickIds[i] == possibleBrickId) {
                return;
            }
            if(possibleTargetBrickIds[i] == UNINIT_STATE_32)
            {
                possibleTargetBrickIds[i] = possibleBrickId;
                numberOfPossibleBricks++;
            }
        }
    }

    uint32_t getPossibleBrick()
    {
        const uint32_t result = possibleTargetBrickIds[0];
        for(uint32_t i = 0; i < numberOfPossibleBricks; i++) {
            possibleTargetBrickIds[i] = possibleTargetBrickIds[i+1];
        }
        return result;
    }

    // total size: 4096 Bytes
};

#endif // HANAMI_BRICK_H
