/**
 * @file        permission.h
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

#ifndef PERMISSION_H
#define PERMISSION_H

#include <libKitsunemimiCommon/threading/event.h>
#include <libKitsunemimiHanamiCommon/structs.h>
#include <libKitsunemimiCommon/logger.h>

namespace Kitsunemimi {
class JsonItem;
}

namespace Kitsunemimi::Hanami
{
struct BlossomStatus;

bool checkPermission(DataMap &context,
                     const std::string &token,
                     Kitsunemimi::Hanami::BlossomStatus &status,
                     const bool skipPermission,
                     Kitsunemimi::ErrorContainer &error);

bool getPermission(JsonItem &parsedResult,
                   const std::string &token,
                   Kitsunemimi::Hanami::BlossomStatus &status,
                   Kitsunemimi::ErrorContainer &error);

}

#endif // PERMISSION_H
