/**
 * @file        runtime_validation.cpp
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

#include "runtime_validation.h"

#include <api/endpoint_processing/items/item_methods.h>
#include <api/endpoint_processing/blossom.h>

const std::string
createErrorMessage(const std::string &name,
                   const FieldType fieldType)
{
    std::string err = "Value-validation failed, because a item '" + name + "'  has the false type:";
    switch(fieldType)
    {
        case SAKURA_UNDEFINED_TYPE:
            break;
        case SAKURA_INT_TYPE:
            err.append(" Exprect int-type.");
            break;
        case SAKURA_FLOAT_TYPE:
            err.append(" Exprect float-type.");
            break;
        case SAKURA_BOOL_TYPE:
            err.append(" Exprect bool-type.");
            break;
        case SAKURA_STRING_TYPE:
            err.append(" Exprect string-type.");
            break;
        case SAKURA_ARRAY_TYPE:
            err.append(" Exprect array-type.");
            break;
        case SAKURA_MAP_TYPE:
            err.append(" Exprect map-type.");
            break;
    }

    return err;
}

/**
 * @brief check valure-types of the blossom-input and -output
 *
 * @param defs definitions to check against
 * @param values map of the input or output-values to validate
 * @param ioType select input- or output-values
 * @param errorMessage reference for error-output
 *
 * @return true, if everything match, else false
 */
bool
checkBlossomValues(const std::map<std::string, FieldDef> &defs,
                   const Kitsunemimi::DataMap &values,
                   const FieldDef::IO_ValueType ioType,
                   std::string &errorMessage)
{
    for(const auto& [name, field] : defs)
    {
        if(field.ioType != ioType) {
            continue;
        }

        Kitsunemimi::DataItem* item = values.get(name);

        if(item != nullptr)
        {
            // check type
            if(checkType(item, field.fieldType) == false)
            {
                errorMessage = createErrorMessage(name, field.fieldType);
                return false;
            }

            // check regex
            if(field.regex.size() > 0)
            {
                const std::regex re("^" + field.regex + "$");
                if(std::regex_match(item->toValue()->getString(), re) == false)
                {
                    errorMessage= "Given item '"
                                  + name
                                  + "' doesn't match with regex \"^"
                                  + field.regex
                                  + "$\"";
                    return false;
                }
            }

            // check value border
            if(field.upperBorder != 0
                    || field.lowerBorder != 0)
            {
                if(item->isIntValue())
                {
                    const long value = item->toValue()->getLong();
                    if(value < field.lowerBorder)
                    {
                        errorMessage = "Given item '"
                                       + name
                                       + "' is smaller than "
                                       + std::to_string(field.lowerBorder);
                        return false;
                    }

                    if(value > field.upperBorder)
                    {
                        errorMessage = "Given item '"
                                       + name
                                       + "' is bigger than "
                                       + std::to_string(field.upperBorder);
                        return false;
                    }
                }

                if(item->isStringValue())
                {
                    const long length = item->toValue()->getString().size();
                    if(length < field.lowerBorder)
                    {
                        errorMessage = "Given item '"
                                       + name
                                       + "' is shorter than "
                                       + std::to_string(field.lowerBorder)
                                       + " characters";
                        return false;
                    }

                    if(length > field.upperBorder)
                    {
                        errorMessage = "Given item '"
                                       + name
                                       + "' is longer than "
                                       + std::to_string(field.upperBorder)
                                       + " characters";
                        return false;
                    }
                }
            }

            // check match
            if(field.match != nullptr)
            {
                if(field.match->toString() != item->toString())
                {
                    errorMessage = "Item '"
                                   + name
                                   + "' doesn't match the the expected value:\n   ";
                    errorMessage.append(field.match->toString());
                    errorMessage.append("\nbut has value:\n   ");
                    errorMessage.append(item->toString());
                    return false;
                }
            }
        }
    }

    return true;
}

/**
 * @brief Check type of an item with the registered field
 *
 * @param item item to check
 * @param fieldType field-type to compare
 *
 * @return true, if match, else false
 */
bool
checkType(Kitsunemimi::DataItem* item,
          const FieldType fieldType)
{
    if(item->getType() == Kitsunemimi::DataItem::ARRAY_TYPE
            && fieldType == SAKURA_ARRAY_TYPE)
    {
        return true;
    }

    if(item->getType() == Kitsunemimi::DataItem::MAP_TYPE
            && fieldType == SAKURA_MAP_TYPE)
    {
        return true;
    }

    if(item->getType() == Kitsunemimi::DataItem::VALUE_TYPE)
    {
        Kitsunemimi::DataValue* value = item->toValue();
        if(value->getValueType() == Kitsunemimi::DataItem::INT_TYPE
                && fieldType == SAKURA_INT_TYPE)
        {
            return true;
        }
        if(value->getValueType() == Kitsunemimi::DataItem::FLOAT_TYPE
                && fieldType == SAKURA_FLOAT_TYPE)
        {
            return true;
        }
        if(value->getValueType() == Kitsunemimi::DataItem::BOOL_TYPE
                && fieldType == SAKURA_BOOL_TYPE)
        {
            return true;
        }
        if(value->getValueType() == Kitsunemimi::DataItem::STRING_TYPE
                && fieldType == SAKURA_STRING_TYPE)
        {
            return true;
        }
    }

    return false;
}

