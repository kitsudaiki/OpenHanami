/**
 * @file        runtime_validation.h
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

#ifndef KITSUNEMIMI_SAKURA_LANG_RUNTIME_VALIDATION_H
#define KITSUNEMIMI_SAKURA_LANG_RUNTIME_VALIDATION_H

#include <regex>

#include <libKitsunemimiCommon/items/data_items.h>
#include <libKitsunemimiCommon/logger.h>

#include <items/value_items.h>
#include <libKitsunemimiHanamiNetwork/blossom.h>

namespace Kitsunemimi::Hanami
{
class ValueItemMap;

bool checkBlossomValues(const std::map<std::string, FieldDef> &defs,
                        const DataMap &values,
                        const FieldDef::IO_ValueType ioType,
                        std::string &errorMessage);

bool checkType(DataItem* item,
               const FieldType fieldType);

}

#endif // KITSUNEMIMI_SAKURA_LANG_RUNTIME_VALIDATION_H
