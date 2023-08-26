/**
 * @file        check_data_set.cpp
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

#include "check_data_set.h"

#include <hanami_root.h>
#include <database/request_result_table.h>
#include <database/data_set_table.h>

#include <libKitsunemimiHanamiFiles/data_set_files/data_set_file.h>

#include <libKitsunemimiJson/json_item.h>
#include <libKitsunemimiCommon/methods/file_methods.h>
#include <libKitsunemimiCommon/buffer/data_buffer.h>
#include <libKitsunemimiCommon/files/text_file.h>
#include <libKitsunemimiCommon/files/binary_file.h>
#include <libKitsunemimiConfig/config_handler.h>

CheckDataSet::CheckDataSet()
    : Blossom("Compare a list of values with a data-set to check correctness.")
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("result_uuid",
                       SAKURA_STRING_TYPE,
                       true,
                       "UUID of the data-set to compare to.");
    assert(addFieldRegex("result_uuid", UUID_REGEX));

    registerInputField("data_set_uuid",
                       SAKURA_STRING_TYPE,
                       true,
                       "UUID of the data-set to compare to.");
    assert(addFieldRegex("data_set_uuid", UUID_REGEX));

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("correctness",
                        SAKURA_FLOAT_TYPE,
                        "Correctness of the values compared to the data-set.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
CheckDataSet::runTask(BlossomIO &blossomIO,
                      const Kitsunemimi::DataMap &context,
                      BlossomStatus &status,
                      Kitsunemimi::ErrorContainer &error)
{
    const std::string resultUuid = blossomIO.input.get("result_uuid").getString();
    const std::string dataUuid = blossomIO.input.get("data_set_uuid").getString();
    const UserContext userContext(context);

    // get result
    // check if request-result exist within the table
    Kitsunemimi::JsonItem result;
    if(RequestResultTable::getInstance()->getRequestResult(result,
                                                           resultUuid,
                                                           userContext,
                                                           error,
                                                           true) == false)
    {
        status.errorMessage = "Request-result with UUID '" + resultUuid + "' not found.";
        status.statusCode = NOT_FOUND_RTYPE;
        error.addMeesage(status.errorMessage);
        return false;
    }

    // get data-info from database
    if(DataSetTable::getInstance()->getDataSet(blossomIO.output,
                                               dataUuid,
                                               userContext,
                                               error,
                                               true) == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // get file information
    const std::string location = blossomIO.output.get("location").getString();

    Kitsunemimi::DataBuffer buffer;
    DataSetFile::DataSetHeader dataSetHeader;
    DataSetFile::ImageTypeHeader imageTypeHeader;
    Kitsunemimi::BinaryFile file(location);

    // read data-set-header
    if(file.readCompleteFile(buffer, error) == false)
    {
        error.addMeesage("Failed to read data-set-header from file '" + location + "'");
        return false;
    }

    // prepare values
    uint64_t correctValues = 0;
    uint64_t dataPos = sizeof(DataSetFile::DataSetHeader) + sizeof(DataSetFile::ImageTypeHeader);
    const uint8_t* u8Data = static_cast<const uint8_t*>(buffer.data);
    memcpy(&dataSetHeader, buffer.data, sizeof(DataSetFile::DataSetHeader));
    memcpy(&imageTypeHeader,
           &u8Data[sizeof(DataSetFile::DataSetHeader)],
           sizeof(DataSetFile::ImageTypeHeader));
    const uint64_t lineOffset = imageTypeHeader.numberOfInputsX * imageTypeHeader.numberOfInputsY;
    const uint64_t lineSize = (imageTypeHeader.numberOfInputsX * imageTypeHeader.numberOfInputsY)
                              + imageTypeHeader.numberOfOutputs;
    const float* content = reinterpret_cast<const float*>(&u8Data[dataPos]);

    // iterate over all values and check
    Kitsunemimi::DataArray* compareData = result.get("data").getItemContent()->toArray();
    for(uint64_t i = 0; i < compareData->size(); i++)
    {
        const uint64_t actualPos = (i * lineSize) + lineOffset;
        const uint64_t checkVal = compareData->get(i)->toValue()->getInt();
        if(content[actualPos + checkVal] > 0.0f) {
            correctValues++;
        }
    }

    // add result to output
    const float correctness = (100.0f / static_cast<float>(compareData->size()))
                              * static_cast<float>(correctValues);
    blossomIO.output.deleteContent();
    blossomIO.output.insert("correctness", correctness);

    return true;
}
