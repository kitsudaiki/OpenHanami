/**
 * @file        io.cpp
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

#ifndef IO_H
#define IO_H

#include <string>
#include <vector>
#include <stdint.h>
#include <chrono>

#include <libKitsunemimiCommon/logger.h>

namespace HanamiAI
{
class WebsocketClient;

bool train(WebsocketClient* wsClient,
           std::vector<float> &inputValues,
           std::vector<float> &shouldValues,
           Kitsunemimi::ErrorContainer &error);

float* request(WebsocketClient* wsClient,
               std::vector<float> &inputValues,
               uint64_t &numberOfOutputValues,
               Kitsunemimi::ErrorContainer &error);

bool train(WebsocketClient* wsClient,
           float* inputValues,
           const uint64_t numberOfInputValues,
           float* shouldValues,
           const uint64_t numberOfShouldValues,
           Kitsunemimi::ErrorContainer &error);

float* request(WebsocketClient* wsClient,
               float* inputData,
               const uint64_t numberOfInputValues,
               uint64_t &numberOfOutputValues,
               Kitsunemimi::ErrorContainer &error);

} // namespace HanamiAI

#endif // IO_H
