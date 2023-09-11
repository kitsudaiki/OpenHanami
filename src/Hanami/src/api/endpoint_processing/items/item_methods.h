/**
 * @file        item_methods.h
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

#ifndef HANAMI_LANG_ITEM_METHODS_H
#define HANAMI_LANG_ITEM_METHODS_H

#include <vector>
#include <string>

#include <hanami_common/logger.h>

#include <api/endpoint_processing/items/sakura_items.h>

struct BlossomIO;

// override functions
enum OverrideType
{
    ALL,
    ONLY_EXISTING,
    ONLY_NON_EXISTING
};

void overrideItems(json &original,
                   const json &override,
                   OverrideType type);

// error-output
void createError(const BlossomItem &blossomItem,
                 const std::string &blossomPath,
                 const std::string &errorLocation,
                 Hanami::ErrorContainer &error,
                 const std::string &possibleSolution = "");
void createError(const BlossomIO &blossomIO,
                 const std::string &errorLocation,
                 Hanami::ErrorContainer &error,
                 const std::string &possibleSolution = "");

#endif // HANAMI_LANG_ITEM_METHODS_H
