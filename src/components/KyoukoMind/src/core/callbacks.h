/**
 * @file        callbacks.h
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

#ifndef KYOUKOMIND_CALLBACKS_H
#define KYOUKOMIND_CALLBACKS_H

#include <kyouko_root.h>

#include <libKitsunemimiHanamiNetwork/hanami_messaging.h>

#include <libKitsunemimiCommon/logger.h>

/**
 * @brief clientDataCallback
 * @param data
 * @param dataSize
 */
void
clientDataCallback(Kitsunemimi::Sakura::Session*,
                   const void* data,
                   const uint64_t dataSize)
{
    const char* charData = static_cast<const char*>(data);
    const std::string text(charData, dataSize);
    LOG_WARNING("client-text: " + text);
}

void genericCallback(Kitsunemimi::Sakura::Session*,
                     const uint32_t,
                     void*,
                     const uint64_t,
                     const uint64_t)
{

}

#endif // KYOUKOMIND_CALLBACKS_H
