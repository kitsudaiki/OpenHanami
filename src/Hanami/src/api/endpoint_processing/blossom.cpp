/**
 * @file        blossom.cpp
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

#include <api/endpoint_processing/blossom.h>

#include <api/endpoint_processing/items/item_methods.h>
#include <api/endpoint_processing/runtime_validation.h>
#include <common.h>

#include <libKitsunemimiCommon/logger.h>

/**
 * @brief constructor
 */
Blossom::Blossom(const std::string &comment, const bool requiresToken)
    : comment(comment),
      requiresAuthToken(requiresToken)
{}

/**
 * @brief destructor
 */
Blossom::~Blossom() {}

/**
 * @brief register input field for validation of incoming messages
 *
 * @param name name of the filed to identifiy value
 * @param fieldType type for value-validation
 *
 * @return reference to entry for further updates
 */
FieldDef&
Blossom::registerInputField(const std::string &name,
                            const FieldType fieldType)
{
    errorCodes.push_back(BAD_REQUEST_RTYPE);
    auto ret = m_inputValidationMap.try_emplace(name, FieldDef(FieldDef::INPUT_TYPE, fieldType));
    return ret.first->second;
}

/**
 * @brief register output field for validation of incoming messages
 *
 * @param name name of the filed to identifiy value
 * @param fieldType type for value-validation
 *
 * @return reference to entry for further updates
 */
FieldDef&
Blossom::registerOutputField(const std::string &name,
                             const FieldType fieldType)
{
    auto ret = m_outputValidationMap.try_emplace(name, FieldDef(FieldDef::OUTPUT_TYPE, fieldType));
    return ret.first->second;
}

/**
 * @brief get a pointer to the input-validation-map
 *
 * @return pointer to validation-map
 */
const std::map<std::string, FieldDef>*
Blossom::getInputValidationMap() const
{
    return &m_inputValidationMap;
}


/**
 * @brief get a pointer to the output-validation-map
 *
 * @return pointer to validation-map
 */
const std::map<std::string, FieldDef>*
Blossom::getOutputValidationMap() const
{
    return &m_outputValidationMap;
}

/**
 * @brief fill with default-values
 *
 * @param values input-values to fill
 */
void
Blossom::fillDefaultValues(Kitsunemimi::DataMap &values)
{
    for(const auto& [name, field] : m_inputValidationMap)
    {
        if(field.defaultVal != nullptr)
        {
            Kitsunemimi::DataItem* tempItem = field.defaultVal->copy();
            if(values.insert(name, tempItem, false) == false) {
                delete tempItem;
            }
        }
    }
}

/**
 * @brief execute blossom
 *
 * @param blossomIO leaf-object for values-handling while processing
 * @param context const-map with global accasible values
 * @param status reference for status-output
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
Blossom::growBlossom(BlossomIO &blossomIO,
                     const Kitsunemimi::DataMap* context,
                     BlossomStatus &status,
                     Kitsunemimi::ErrorContainer &error)
{
    LOG_DEBUG("runTask " + blossomIO.blossomName);

    // set default-values
    fillDefaultValues(*blossomIO.input.getItemContent()->toMap());
    std::string errorMessage;

    // validate input
    if(checkBlossomValues(m_inputValidationMap,
                          *blossomIO.input.getItemContent()->toMap(),
                          FieldDef::INPUT_TYPE,
                          errorMessage) == false)
    {
        error.addMeesage(errorMessage);
        status.errorMessage = errorMessage;
        status.statusCode = 400;
        return false;
    }

    // handle result
    if(runTask(blossomIO, *context, status, error) == false)
    {
        createError(blossomIO, "blossom execute", error);
        return false;
    }

    // validate output
    if(checkBlossomValues(m_outputValidationMap,
                          *blossomIO.output.getItemContent()->toMap(),
                          FieldDef::OUTPUT_TYPE,
                          errorMessage) == false)
    {
        error.addMeesage(errorMessage);
        status.errorMessage = errorMessage;
        status.statusCode = 500;
        return false;
    }

    return true;
}

/**
 * @brief validate given input with the required and allowed values of the selected blossom
 *
 * @param input given input values
 * @param valueType say if input or output should be checked
 * @param errorMessage reference for error-output
 *
 * @return true, if successful, else false
 */
bool
Blossom::validateFieldsCompleteness(const DataMap &input,
                                    const std::map<std::string, FieldDef> &validationMap,
                                    const FieldDef::IO_ValueType valueType,
                                    std::string &errorMessage)
{
    if(allowUnmatched == false)
    {
        // check if all keys in the values of the blossom-item also exist in the required-key-list
        for(const auto& [name, _] : input.map)
        {
            if(validationMap.find(name) == validationMap.end())
            {
                // build error-output
                errorMessage = "Validation failed, because item '"
                               + name
                               + "' is not in the list of allowed keys";
                return false;
            }
        }
    }

    // check that all keys in the required keys are also in the values of the blossom-item
    for(const auto& [name, field] : validationMap)
    {
        if(field.isRequired == true
                && field.ioType == valueType)
        {
            // search for values
            if(input.contains(name) == false)
            {
                errorMessage = "Validation failed, because variable '"
                               + name
                               + "' is required, but is not set.";
                return false;
            }
        }
    }

    return true;
}

/**
 * @brief validate given input with the required and allowed values of the selected blossom
 *
 * @param blossomItem blossom-item with given values
 * @param filePath file-path where the blossom belongs to, only used for error-output
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
Blossom::validateInput(BlossomItem &blossomItem,
                       const std::map<std::string, FieldDef> &validationMap,
                       const std::string &filePath,
                       Kitsunemimi::ErrorContainer &error)
{
    std::map<std::string, FieldDef::IO_ValueType> compareMap;
    getCompareMap(compareMap, blossomItem.values);

    if(allowUnmatched == false)
    {
        // check if all keys in the values of the blossom-item also exist in the required-key-list
        for(const auto& [name, field] : compareMap)
        {
            if(validationMap.find(name) == validationMap.end())
            {
                // build error-output
                error.addMeesage("item '"
                                 + name
                                 + "' is not in the list of allowed keys");
                createError(blossomItem, filePath, "validator", error);
                return false;
            }
        }
    }

    // check that all keys in the required keys are also in the values of the blossom-item
    for(const auto& [name, field] : validationMap)
    {
        if(field.isRequired == true)
        {
            // search for values
            auto compareIt = compareMap.find(name);
            if(compareIt != compareMap.end())
            {
                if(field.ioType != compareIt->second)
                {
                    error.addMeesage("item '"
                                     + name
                                     + "' has not the correct input/output type");
                    createError(blossomItem, filePath, "validator", error);
                    return false;
                }
            }
            else
            {
                error.addMeesage("item '"
                                 + name
                                 + "' is required, but is not set.");
                createError(blossomItem, filePath, "validator", error);
                return false;
            }
        }
    }

    return true;
}

/**
 * @brief get map for comparism in validator
 *
 * @param compareMap reference for the resulting map
 * @param value-map to compare
 */
void
Blossom::getCompareMap(std::map<std::string, FieldDef::IO_ValueType> &compareMap,
                       const ValueItemMap &valueMap)
{
    // copy items
    for(const auto& [id, item] : valueMap.m_valueMap)
    {
        if(item.type == ValueItem::INPUT_PAIR_TYPE) {
            compareMap.emplace(id, FieldDef::INPUT_TYPE);
        }

        if(item.type == ValueItem::OUTPUT_PAIR_TYPE) {
            compareMap.emplace(item.item->toString(), FieldDef::OUTPUT_TYPE);
        }
    }

    // copy child-maps
    for(const auto& [id, _] : valueMap.m_childMaps)
    {
        compareMap.emplace(id, FieldDef::INPUT_TYPE);
    }
}
