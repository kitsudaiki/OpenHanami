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
#include <api/endpoint_processing/runtime_validation.h>
#include <hanami_common/logger.h>

/**
 * @brief constructor
 */
Blossom::Blossom(const std::string& comment, const bool requiresToken)
    : comment(comment), requiresAuthToken(requiresToken)
{
    if (requiresToken) {
        errorCodes.push_back(UNAUTHORIZED_RTYPE);
    }
}

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
Blossom::registerInputField(const std::string& name, const FieldType fieldType)
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
Blossom::registerOutputField(const std::string& name, const FieldType fieldType)
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
Blossom::fillDefaultValues(json& values)
{
    for (const auto& [name, field] : m_inputValidationMap) {
        if (field.defaultVal != nullptr) {
            if (values.contains(name) == false) {
                values[name] = field.defaultVal;
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
Blossom::growBlossom(BlossomIO& blossomIO,
                     const json& context,
                     BlossomStatus& status,
                     Hanami::ErrorContainer& error)
{
    LOG_DEBUG("runTask " + blossomIO.blossomName);

    // set default-values
    fillDefaultValues(blossomIO.input);
    std::string errorMessage;

    // validate input
    if (checkBlossomValues(
            m_inputValidationMap, blossomIO.input, FieldDef::INPUT_TYPE, errorMessage)
        == false)
    {
        status.errorMessage = errorMessage;
        status.statusCode = BAD_REQUEST_RTYPE;
        LOG_DEBUG(errorMessage);
        return false;
    }

    // handle result
    if (runTask(blossomIO, context, status, error) == false) {
        if (status.statusCode == INTERNAL_SERVER_ERROR_RTYPE) {
            createError(blossomIO, "blossom execute", error);
        }
        return false;
    }

    // validate output
    if (checkBlossomValues(
            m_outputValidationMap, blossomIO.output, FieldDef::OUTPUT_TYPE, errorMessage)
        == false)
    {
        error.addMessage(errorMessage);
        status.errorMessage = errorMessage;
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
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
Blossom::validateFieldsCompleteness(const json& input,
                                    const std::map<std::string, FieldDef>& validationMap,
                                    const FieldDef::IO_ValueType valueType,
                                    std::string& errorMessage)
{
    if (allowUnmatched == false) {
        // check if all keys in the values of the blossom-item also exist in the required-key-list
        for (const auto& [name, _] : input.items()) {
            if (validationMap.find(name) == validationMap.end()) {
                // build error-output
                errorMessage = "Validation failed, because item '" + name
                               + "' is not in the list of allowed keys";
                return false;
            }
        }
    }

    // check that all keys in the required keys are also in the values of the blossom-item
    for (const auto& [name, field] : validationMap) {
        if (field.isRequired == true && field.ioType == valueType) {
            // search for values
            if (input.contains(name) == false) {
                errorMessage = "Validation failed, because variable '" + name
                               + "' is required, but is not set.";
                return false;
            }
        }
    }

    return true;
}

/**
 * @brief create an error-output
 *
 * @param blossomIO blossom-item with information of the error-location
 * @param errorLocation location where the error appeared
 * @param error reference for error-output
 */
void
Blossom::createError(const BlossomIO& blossomIO,
                     const std::string& errorLocation,
                     Hanami::ErrorContainer& error)
{
    Hanami::TableItem errorOutput;
    // initialize error-output
    errorOutput.addColumn("Field");
    errorOutput.addColumn("Value");

    if (errorLocation.size() > 0) {
        errorOutput.addRow(std::vector<std::string>{"location", errorLocation});
    }

    if (blossomIO.blossomType.size() > 0) {
        errorOutput.addRow(std::vector<std::string>{"blossom-type", blossomIO.blossomType});
    }
    if (blossomIO.blossomGroupType.size() > 0) {
        errorOutput.addRow(
            std::vector<std::string>{"blossom-group-type", blossomIO.blossomGroupType});
    }
    if (blossomIO.blossomName.size() > 0) {
        errorOutput.addRow(std::vector<std::string>{"blossom-name", blossomIO.blossomName});
    }

    error.addMessage("Error in location: \n" + errorOutput.toString(200, true));
}
