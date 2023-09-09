/**
 * @file        value_items.h
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

#ifndef HANAMI_LANG_VALUE_ITEMS_H
#define HANAMI_LANG_VALUE_ITEMS_H

#include <string>
#include <vector>

#include <api/endpoint_processing/blossom.h>

//==================================================================================================
// FunctionItem
//==================================================================================================
struct ValueItem;

struct FunctionItem
{
    std::string type = "";
    std::vector<ValueItem> arguments;
};

//==================================================================================================
// ValueItem
//==================================================================================================
struct ValueItem
{
    enum ValueType
    {
        UNDEFINED_PAIR_TYPE = 0,
        INPUT_PAIR_TYPE = 1,
        OUTPUT_PAIR_TYPE = 2,
        COMPARE_EQUAL_PAIR_TYPE = 3,
        COMPARE_UNEQUAL_PAIR_TYPE = 4,
    };

    json item = nullptr;
    ValueType type = INPUT_PAIR_TYPE;
    bool isIdentifier = false;
    std::string comment = "";
    FieldType fieldType = SAKURA_UNDEFINED_TYPE;
    std::vector<FunctionItem> functions;

    ValueItem() {}

    ValueItem(const ValueItem &other)
    {
        item = other.item;
        type = other.type;
        isIdentifier = other.isIdentifier;
        functions = other.functions;
        fieldType = other.fieldType;
        comment = other.comment;
    }

    ~ValueItem() {}

    ValueItem &operator=(const ValueItem &other)
    {
        if(this != &other)
        {
            this->item = other.item;
            this->type = other.type;
            this->isIdentifier = other.isIdentifier;
            this->functions = other.functions;
            this->fieldType = other.fieldType;
            this->comment = other.comment;
        }
        return *this;
    }
};

#endif // HANAMI_LANG_VALUE_ITEMS_H
