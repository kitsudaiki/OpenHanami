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

#include <hanami_common/items/data_items.h>
#include <hanami_common/logger.h>

#include <api/endpoint_processing/items/value_items.h>
#include <api/endpoint_processing//blossom.h>

class ValueItemMap;

bool checkBlossomValues(const std::map<std::string, FieldDef> &defs,
                        const Hanami::DataMap &values,
                        const FieldDef::IO_ValueType ioType,
                        std::string &errorMessage);

bool checkType(Hanami::DataItem* item,
               const FieldType fieldType);

#endif // KITSUNEMIMI_SAKURA_LANG_RUNTIME_VALIDATION_H
