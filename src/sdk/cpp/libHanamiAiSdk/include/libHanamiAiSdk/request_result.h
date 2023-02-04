/**
 * @file        request_result.h
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

#ifndef KITSUNEMIMI_HANAMISDK_REQUEST_RESULT_H
#define KITSUNEMIMI_HANAMISDK_REQUEST_RESULT_H

#include <libKitsunemimiCommon/logger.h>

namespace HanamiAI
{

bool getRequestResult(std::string &result,
                      const std::string &userId,
                      Kitsunemimi::ErrorContainer &error);

bool listRequestResult(std::string &result,
                       Kitsunemimi::ErrorContainer &error);

bool deleteRequestResult(std::string &result,
                         const std::string &userId,
                         Kitsunemimi::ErrorContainer &error);

} // namespace HanamiAI

#endif // KITSUNEMIMI_HANAMISDK_REQUEST_RESULT_H
