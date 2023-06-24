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
#include <libKitsunemimiCommon/logger.h>
#include <api/endpoint_processing/runtime_validation.h>

/**
 * @brief constructor
 */
Blossom::Blossom(const std::string &comment)
    : comment(comment) {}

/**
 * @brief destructor
 */
Blossom::~Blossom() {}

/**
 * @brief register input field for validation of incoming messages
 *
 * @param name name of the filed to identifiy value
 * @param fieldType type for value-validation
 * @param required false, to make field optional, true to make it required
 * @param comment additional comment to describe the content of the field
 *
 * @return false, if already name already registered, else true
 */
bool
Blossom::registerInputField(const std::string &name,
                            const FieldType fieldType,
                            const bool required,
                            const std::string &comment)
{
    std::map<std::string, FieldDef>::const_iterator defIt;
    defIt = m_inputValidationMap.find(name);
    if(defIt != m_inputValidationMap.end()) {
        return false;
    }

    m_inputValidationMap.emplace(name, FieldDef(FieldDef::INPUT_TYPE, fieldType, required, comment));

    return true;
}

/**
 * @brief register output field for validation of incoming messages
 *
 * @param name name of the filed to identifiy value
 * @param fieldType type for value-validation
 * @param comment additional comment to describe the content of the field
 *
 * @return false, if already name already registered, else true
 */
bool
Blossom::registerOutputField(const std::string &name,
                             const FieldType fieldType,
                             const std::string &comment)
{
    std::map<std::string, FieldDef>::const_iterator defIt;
    defIt = m_outputValidationMap.find(name);
    if(defIt != m_outputValidationMap.end()) {
        return false;
    }

    m_outputValidationMap.emplace(name, FieldDef(FieldDef::OUTPUT_TYPE, fieldType, false, comment));

    return true;
}

/**
 * @brief add match-value for a specific field for static expected outputs
 *
 * @param name name of the filed to identifiy value
 * @param match value, which should match in the validation
 *
 * @return false, if field doesn't exist, else true
 */
bool
Blossom::addFieldMatch(const std::string &name,
                       Kitsunemimi::DataItem* match)
{
    std::map<std::string, FieldDef>::iterator defIt;
    defIt = m_outputValidationMap.find(name);
    if(defIt != m_outputValidationMap.end())
    {
        // delete old entry
        if(defIt->second.match != nullptr) {
            delete defIt->second.match;
        }

        defIt->second.match = match;
        return true;
    }

    return false;
}

/**
 * @brief add default-value for a specific field (only works if the field is NOT required)
 *
 * @param name name of the filed to identifiy value
 * @param defaultValue default-value for a field
 *
 * @return false, if field doesn't exist, else true
 */
bool
Blossom::addFieldDefault(const std::string &name,
                         Kitsunemimi::DataItem* defaultValue)
{
    std::map<std::string, FieldDef>::iterator defIt;
    defIt = m_inputValidationMap.find(name);
    if(defIt != m_inputValidationMap.end())
    {
        // make sure, that it is not required
        if(defIt->second.isRequired) {
            return false;
        }

        // delete old entry
        if(defIt->second.defaultVal != nullptr) {
            delete defIt->second.defaultVal;
        }

        defIt->second.defaultVal = defaultValue;
        return true;
    }

    return false;
}

/**
 * @brief add regex to check string for special styling
 *
 * @param name name of the filed to identifiy value
 * @param regex regex-string
 *
 * @return false, if field doesn't exist or a string-type, else true
 */
bool
Blossom::addFieldRegex(const std::string &name,
                       const std::string &regex)
{
    std::map<std::string, FieldDef>::iterator defIt;
    defIt = m_inputValidationMap.find(name);
    if(defIt != m_inputValidationMap.end())
    {
        // make sure, that it is a string-type
        if(defIt->second.fieldType != SAKURA_STRING_TYPE) {
            return false;
        }

        defIt->second.regex = regex;

        return true;
    }

    return false;
}

/**
 * @brief add lower and upper border for int and string values
 *
 * @param name name of the filed to identifiy value
 * @param lowerBorder lower value or length border
 * @param upperBorder upper value or length border
 *
 * @return false, if field doesn't exist or not matching requirements, else true
 */
bool
Blossom::addFieldBorder(const std::string &name,
                        const long lowerBorder,
                        const long upperBorder)
{
    std::map<std::string, FieldDef>::iterator defIt;
    defIt = m_inputValidationMap.find(name);
    if(defIt != m_inputValidationMap.end())
    {
        // make sure, that it is an int- or string-type
        if(defIt->second.fieldType != SAKURA_STRING_TYPE
                && defIt->second.fieldType != SAKURA_INT_TYPE)
        {
            return false;
        }

        defIt->second.lowerBorder = lowerBorder;
        defIt->second.upperBorder = upperBorder;

        return true;
    }

    return false;
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
    std::map<std::string, FieldDef>::const_iterator defIt;
    for(defIt = m_inputValidationMap.begin();
        defIt != m_inputValidationMap.end();
        defIt++)
    {
        if(defIt->second.defaultVal != nullptr) {
            values.insert(defIt->first, defIt->second.defaultVal->copy(), false);
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
                     Kitsunemimi::Hanami::BlossomStatus &status,
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
        std::map<std::string, Kitsunemimi::DataItem*>::const_iterator compareIt;
        for(compareIt = input.map.begin();
            compareIt != input.map.end();
            compareIt++)
        {
            std::map<std::string, FieldDef>::const_iterator defIt;
            defIt = validationMap.find(compareIt->first);
            if(defIt == validationMap.end())
            {
                // build error-output
                errorMessage = "Validation failed, because item '"
                               + compareIt->first
                               + "' is not in the list of allowed keys";
                return false;
            }
        }
    }

    // check that all keys in the required keys are also in the values of the blossom-item
    std::map<std::string, FieldDef>::const_iterator defIt;
    for(defIt = validationMap.begin();
        defIt != validationMap.end();
        defIt++)
    {
        if(defIt->second.isRequired == true
                && defIt->second.ioType == valueType)
        {
            // search for values
            if(input.contains(defIt->first) == false)
            {
                errorMessage = "Validation failed, because variable '"
                               + defIt->first
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
        std::map<std::string, FieldDef::IO_ValueType>::const_iterator compareIt;
        for(compareIt = compareMap.begin();
            compareIt != compareMap.end();
            compareIt++)
        {
            std::map<std::string, FieldDef>::const_iterator defIt;
            defIt = validationMap.find(compareIt->first);
            if(defIt == validationMap.end())
            {
                // build error-output
                error.addMeesage("item '"
                                 + compareIt->first
                                 + "' is not in the list of allowed keys");
                createError(blossomItem, filePath, "validator", error);
                return false;
            }
        }
    }

    // check that all keys in the required keys are also in the values of the blossom-item
    std::map<std::string, FieldDef>::const_iterator defIt;
    for(defIt = validationMap.begin();
        defIt != validationMap.end();
        defIt++)
    {
        if(defIt->second.isRequired == true)
        {
            // search for values
            std::map<std::string, FieldDef::IO_ValueType>::const_iterator compareIt;
            compareIt = compareMap.find(defIt->first);
            if(compareIt != compareMap.end())
            {
                if(defIt->second.ioType != compareIt->second)
                {
                    error.addMeesage("item '"
                                     + defIt->first
                                     + "' has not the correct input/output type");
                    createError(blossomItem, filePath, "validator", error);
                    return false;
                }
            }
            else
            {
                error.addMeesage("item '"
                                 + defIt->first
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
    for(auto const& [id, item] : valueMap.m_valueMap)
    {
        if(item.type == ValueItem::INPUT_PAIR_TYPE) {
            compareMap.emplace(id, FieldDef::INPUT_TYPE);
        }

        if(item.type == ValueItem::OUTPUT_PAIR_TYPE) {
            compareMap.emplace(item.item->toString(), FieldDef::OUTPUT_TYPE);
        }
    }

    // copy child-maps
    std::map<std::string, ValueItemMap*>::const_iterator itChilds;
    for(itChilds = valueMap.m_childMaps.begin();
        itChilds != valueMap.m_childMaps.end();
        itChilds++)
    {
        compareMap.emplace(itChilds->first, FieldDef::INPUT_TYPE);
    }
}
