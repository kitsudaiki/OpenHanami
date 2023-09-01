/**
 * @file        finalize_mnist_data_set.cpp
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

#include "finalize_mnist_data_set.h"

#include <hanami_root.h>
#include <database/data_set_table.h>
#include <core/temp_file_handler.h>

#include <libKitsunemimiHanamiFiles/data_set_files/data_set_file.h>
#include <libKitsunemimiHanamiFiles/data_set_files/image_data_set_file.h>

#include <libKitsunemimiCrypto/common.h>
#include <libKitsunemimiJson/json_item.h>
#include <libKitsunemimiCommon/files/binary_file.h>
#include <libKitsunemimiCommon/methods/file_methods.h>

FinalizeMnistDataSet::FinalizeMnistDataSet()
    : Blossom("Finalize uploaded data-set by checking completeness of the "
              "uploaded and convert into generic format.")
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("uuid", SAKURA_STRING_TYPE)
            .setComment("UUID of the new data-set.")
            .setRegex(UUID_REGEX);

    registerInputField("uuid_input_file", SAKURA_STRING_TYPE)
            .setComment("UUID to identify the file for date upload of input-data.")
            .setRegex(UUID_REGEX);

    registerInputField("uuid_label_file", SAKURA_STRING_TYPE)
            .setComment("UUID to identify the file for date upload of label-data.")
            .setRegex(UUID_REGEX);

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("uuid", SAKURA_STRING_TYPE)
            .setComment("UUID of the new data-set.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
FinalizeMnistDataSet::runTask(BlossomIO &blossomIO,
                              const Kitsunemimi::DataMap &context,
                              BlossomStatus &status,
                              Kitsunemimi::ErrorContainer &error)
{
    const std::string uuid = blossomIO.input.get("uuid").getString();
    const std::string inputUuid = blossomIO.input.get("uuid_input_file").getString();
    const std::string labelUuid = blossomIO.input.get("uuid_label_file").getString();
    const UserContext userContext(context);

    // get location from database
    Kitsunemimi::JsonItem result;
    if(DataSetTable::getInstance()->getDataSet(result, uuid, userContext, error, true) == false)
    {
        status.errorMessage = "Data with uuid '" + uuid + "' not found.";
        status.statusCode = NOT_FOUND_RTYPE;
        return false;
    }

    // read input-data from temp-file
    Kitsunemimi::DataBuffer inputBuffer;
    if(TempFileHandler::getInstance()->getData(inputBuffer, inputUuid) == false)
    {
        status.errorMessage = "Input-data with uuid '" + inputUuid + "' not found.";
        status.statusCode = NOT_FOUND_RTYPE;
        return false;
    }

    // read label from temp-file
    Kitsunemimi::DataBuffer labelBuffer;
    if(TempFileHandler::getInstance()->getData(labelBuffer, labelUuid) == false)
    {
        status.errorMessage = "Label-data with uuid '" + inputUuid + "' not found.";
        status.statusCode = NOT_FOUND_RTYPE;
        return false;
    }

    // write data to file
    if(convertMnistData(result.get("location").getString(),
                        result.get("name").getString().c_str(),
                        inputBuffer,
                        labelBuffer) == false)
    {
        status.statusCode = INTERNAL_SERVER_ERROR_RTYPE;
        error.addMeesage("Failed to convert mnist-data");
        return false;
    }

    // delete temp-files
    TempFileHandler::getInstance()->removeData(inputUuid);
    TempFileHandler::getInstance()->removeData(labelUuid);

    // create output
    blossomIO.output.insert("uuid", uuid);

    return true;
}

/**
 * @brief convert mnist-data into generic format
 *
 * @param filePath path to the resulting file
 * @param name data-set name
 * @param inputBuffer buffer with input-data
 * @param labelBuffer buffer with label-data
 *
 * @return true, if successfull, else false
 */
bool
FinalizeMnistDataSet::convertMnistData(const std::string &filePath,
                                       const std::string &name,
                                       const Kitsunemimi::DataBuffer &inputBuffer,
                                       const Kitsunemimi::DataBuffer &labelBuffer)
{
    Kitsunemimi::ErrorContainer error;
    ImageDataSetFile file(filePath);
    file.type = DataSetFile::IMAGE_TYPE;
    file.name = name;

    // source-data
    const uint64_t dataOffset = 16;
    const uint64_t labelOffset = 8;
    const uint8_t* dataBufferPtr = static_cast<uint8_t*>(inputBuffer.data);
    const uint8_t* labelBufferPtr = static_cast<uint8_t*>(labelBuffer.data);

    // get number of images
    uint32_t numberOfImages = 0;
    numberOfImages |= dataBufferPtr[7];
    numberOfImages |= static_cast<uint32_t>(dataBufferPtr[6]) << 8;
    numberOfImages |= static_cast<uint32_t>(dataBufferPtr[5]) << 16;
    numberOfImages |= static_cast<uint32_t>(dataBufferPtr[4]) << 24;

    // get number of rows
    uint32_t numberOfRows = 0;
    numberOfRows |= dataBufferPtr[11];
    numberOfRows |= static_cast<uint32_t>(dataBufferPtr[10]) << 8;
    numberOfRows |= static_cast<uint32_t>(dataBufferPtr[9]) << 16;
    numberOfRows |= static_cast<uint32_t>(dataBufferPtr[8]) << 24;

    // get number of columns
    uint32_t numberOfColumns = 0;
    numberOfColumns |= dataBufferPtr[15];
    numberOfColumns |= static_cast<uint32_t>(dataBufferPtr[14]) << 8;
    numberOfColumns |= static_cast<uint32_t>(dataBufferPtr[13]) << 16;
    numberOfColumns |= static_cast<uint32_t>(dataBufferPtr[12]) << 24;

    // set information in header
    file.imageHeader.numberOfInputsX = numberOfColumns;
    file.imageHeader.numberOfInputsY = numberOfRows;
    // TODO: read number of labels from file
    file.imageHeader.numberOfOutputs = 10;
    file.imageHeader.numberOfImages = numberOfImages;

    // buffer for values to reduce write-access to file
    const uint64_t lineSize = (numberOfColumns * numberOfRows) * 10;
    const uint32_t segmentSize = lineSize * 10000;
    std::vector<float> segment(segmentSize, 0.0f);
    uint64_t segmentPos = 0;
    uint64_t segmentCounter = 0;

    // init file
    if(file.initNewFile(error) == false) {
        return false;
    }

    // get pictures
    const uint32_t pictureSize = numberOfRows * numberOfColumns;
    double averageVal = 0.0f;
    uint64_t valueCounter = 0;
    float maxVal = 0.0f;

    // copy values of each pixel into the resulting file
    for(uint32_t pic = 0; pic < numberOfImages; pic++)
    {
        // input
        for(uint32_t i = 0; i < pictureSize; i++)
        {
            const uint32_t pos = pic * pictureSize + i + dataOffset;
            segment[segmentPos] = static_cast<float>(dataBufferPtr[pos]);

            // update values for metadata
            averageVal += segment[segmentPos];
            valueCounter++;
            if(maxVal < segment[segmentPos]) {
                maxVal = segment[segmentPos];
            }

            segmentPos++;
        }

        // label
        for(uint32_t i = 0; i < 10; i++)
        {
            segment[segmentPos] = 0.0f;
            segmentPos++;
        }
        const uint32_t label = labelBufferPtr[pic + labelOffset];
        segment[(segmentPos - 10) + label] = 1;

        // write line to file, if segment is full
        if(segmentPos == segmentSize)
        {
            if(file.addBlock(segmentCounter * segmentSize, &segment[0], segmentSize, error) == false) {
                return false;
            }
            segmentPos = 0;
            segmentCounter++;
        }
    }

    // write last incomplete segment to file
    if(segmentPos != 0)
    {
        if(file.addBlock(segmentCounter * segmentSize, &segment[0], segmentPos, error) == false) {
            return false;
        }
    }

    // write additional information to header
    file.imageHeader.avgValue = averageVal / static_cast<double>(valueCounter);
    file.imageHeader.maxValue = maxVal;

    // update header in file for the final number of lines for the case,
    // that there were invalid lines
    if(file.updateHeader(error) == false) {
        return false;
    }

    return true;
}

