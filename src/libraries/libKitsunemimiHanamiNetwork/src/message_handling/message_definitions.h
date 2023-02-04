/**
 * @file        message_definitions.h
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

#ifndef KITSUNEMIMI_HANAMI_MESSAGING_MESSAGE_DEFINITIONS_H
#define KITSUNEMIMI_HANAMI_MESSAGING_MESSAGE_DEFINITIONS_H

#include <stdint.h>
#include <libKitsunemimiHanamiCommon/enums.h>

namespace Kitsunemimi
{
namespace Hanami
{

enum MessageTypes
{
    SAKURA_TRIGGER_MESSAGE = 0,
    SAKURA_GENERIC_MESSAGE = 1,
    RESPONSE_MESSAGE = 4,
};

struct SakuraTriggerHeader
{
    const uint8_t type = SAKURA_TRIGGER_MESSAGE;
    HttpRequestType requestType = GET_TYPE;
    uint32_t idSize = 0;
    uint32_t inputValuesSize = 0;
};

struct SakuraGenericHeader
{
    const uint8_t type = SAKURA_GENERIC_MESSAGE;
    uint32_t subType = 0;
    uint32_t size = 0;
};

struct ResponseHeader
{
    uint8_t type = RESPONSE_MESSAGE;
    bool success = true;
    HttpResponseTypes responseType = OK_RTYPE;
    uint32_t messageSize = 0;
};

}  // namespace Hanami
}  // namespace Kitsunemimi

#endif // KITSUNEMIMI_HANAMI_MESSAGING_MESSAGE_DEFINITIONS_H
