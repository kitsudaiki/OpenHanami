/**
 * @file        finalize_csv_data_set.cpp
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

#include "finalize_csv_data_set.h"

#include <shiori_root.h>
#include <database/data_set_table.h>
#include <core/temp_file_handler.h>
#include <core/data_set_files/data_set_file.h>
#include <core/data_set_files/table_data_set_file.h>

#include <libKitsunemimiHanamiCommon/uuid.h>
#include <libKitsunemimiHanamiCommon/enums.h>
#include <libKitsunemimiHanamiCommon/structs.h>
#include <libKitsunemimiHanamiCommon/defines.h>
#include <libKitsunemimiHanamiNetwork/hanami_messaging.h>

#include <libKitsunemimiCrypto/common.h>
#include <libKitsunemimiJson/json_item.h>
#include <libKitsunemimiCommon/files/binary_file.h>
#include <libKitsunemimiCommon/methods/file_methods.h>
#include <libKitsunemimiCommon/methods/string_methods.h>
#include <libKitsunemimiCommon/methods/vector_methods.h>

using namespace Kitsunemimi::Hanami;

FinalizeCsvDataSet::FinalizeCsvDataSet()
    : Blossom("Finalize uploaded data-set by checking completeness of the "
              "uploaded and convert into generic format.")
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("uuid",
                       SAKURA_STRING_TYPE,
                       true,
                       "UUID of the new data-set.");
    assert(addFieldRegex("uuid", UUID_REGEX));

    registerInputField("uuid_input_file",
                       SAKURA_STRING_TYPE,
                       true,
                       "UUID to identify the file for date upload of input-data.");
    assert(addFieldRegex("uuid_input_file", UUID_REGEX));

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("uuid",
                        SAKURA_STRING_TYPE,
                        "UUID of the new data-set.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
FinalizeCsvDataSet::runTask(BlossomIO &blossomIO,
                            const Kitsunemimi::DataMap &context,
                            BlossomStatus &status,
                            Kitsunemimi::ErrorContainer &error)
{
    const std::string uuid = blossomIO.input.get("uuid").getString();
    const std::string inputUuid = blossomIO.input.get("uuid_input_file").getString();
    const Kitsunemimi::Hanami::UserContext userContext(context);

    // get location from database
    Kitsunemimi::JsonItem result;
    if(ShioriRoot::dataSetTable->getDataSet(result, uuid, userContext, error, true) == false)
    {
        status.errorMessage = "Data with uuid '" + uuid + "' not found.";
        status.statusCode = Kitsunemimi::Hanami::NOT_FOUND_RTYPE;
        return false;
    }

    // read input-data from temp-file
    Kitsunemimi::DataBuffer inputBuffer;
    if(ShioriRoot::tempFileHandler->getData(inputBuffer, inputUuid) == false)
    {
        status.errorMessage = "Input-data with uuid '" + inputUuid + "' not found.";
        status.statusCode = Kitsunemimi::Hanami::NOT_FOUND_RTYPE;
        return false;
    }

    // write data to file
    if(convertCsvData(result.get("location").getString(),
                      result.get("name").getString().c_str(),
                      inputBuffer) == false)
    {
        status.statusCode = Kitsunemimi:: Hanami::INTERNAL_SERVER_ERROR_RTYPE;
        error.addMeesage("Failed to convert csv-data");
        return false;
    }

    // delete temp-files
    ShioriRoot::tempFileHandler->removeData(inputUuid);

    // create output
    blossomIO.output.insert("uuid", uuid);

    return true;
}

void
FinalizeCsvDataSet::convertField(float* segmentPos,
                                 const std::string &cell,
                                 const float lastVal)
{
    // true
    if(cell == "Null"
            || cell == "null"
            || cell == "NULL")
    {
        *segmentPos = lastVal;
    }
    // true
    else if(cell == "True"
            || cell == "true"
            || cell == "TRUE")
    {
        *segmentPos = 1.0f;
    }
    // false
    else if(cell == "False"
            || cell == "false"
            || cell == "FALSE")
    {
        *segmentPos = 0.0f;
    }
    // int/long
    else if(regex_match(cell, std::regex(INT_VALUE_REGEX)))
    {
        *segmentPos = static_cast<float>(std::stoi(cell.c_str()));
    }
    // float/double
    else if(regex_match(cell, std::regex(FLOAT_VALUE_REGEX)))
    {
        *segmentPos = std::strtof(cell.c_str(), NULL);
    }
    else
    {
        // ignore other lines
        *segmentPos = 0.0f;
    }
}

/**
 * @brief convert csv-data into generic format
 *
 * @param filePath path to the resulting file
 * @param name data-set name
 * @param inputBuffer buffer with input-data
 *
 * @return true, if successfull, else false
 */
bool
FinalizeCsvDataSet::convertCsvData(const std::string &filePath,
                                   const std::string &name,
                                   const Kitsunemimi::DataBuffer &inputBuffer)
{
    TableDataSetFile file(filePath);
    file.type = DataSetFile::TABLE_TYPE;
    file.name = name;

    // prepare content-processing
    const std::string stringContent(static_cast<char*>(inputBuffer.data),
                                    inputBuffer.usedBufferSize);

    // buffer for values to reduce write-access to file
    const uint32_t segmentSize = 10000000;
    std::vector<float> segment(segmentSize, 0.0f);
    std::vector<float> lastLine;
    uint64_t segmentPos = 0;
    uint64_t segmentCounter = 0;

    // split content
    std::vector<std::string> lines;
    Kitsunemimi::splitStringByDelimiter(lines, stringContent, '\n');

    bool isHeader = true;

    for(uint64_t lineNum = 0; lineNum < lines.size(); lineNum++)
    {
        const std::string* line = &lines[lineNum];

        // check if the line is relevant to ignore broken lines
        const uint64_t numberOfColumns = std::count(line->begin(), line->end(), ',') + 1;
        if(numberOfColumns == 1) {
            continue;
        }

        // split line
        std::vector<std::string> lineContent;
        Kitsunemimi::splitStringByDelimiter(lineContent, *line, ',');

        if(isHeader)
        {
            file.tableHeader.numberOfColumns = numberOfColumns;
            file.tableHeader.numberOfLines = lines.size();

            for(const std::string &col : lineContent)
            {
                // create and add header-entry
                DataSetFile::TableHeaderEntry entry;
                entry.setName(col);
                file.tableColumns.push_back(entry);
            }
            isHeader = false;

            if(file.initNewFile() == false) {
                return false;
            }

            // this was the max value. While iterating over all lines, this value will be new
            // calculated with the correct value
            file.tableHeader.numberOfLines = 0;
            lastLine = std::vector<float>(numberOfColumns, 0.0f);
        }
        else
        {
            for(uint64_t colNum = 0; colNum < lineContent.size(); colNum++)
            {
                const std::string* cell = &lineContent[colNum];
                if(lastLine.size() > 0)
                {
                    const float lastVal = lastLine[colNum];
                    convertField(&segment[segmentPos], *cell, lastVal);
                }
                else
                {
                    convertField(&segment[segmentPos], *cell, 0.0f);
                }

                lastLine[colNum] = segment[segmentPos];

                // write next segment to file
                segmentPos++;
                if(segmentPos == segmentSize)
                {
                    file.addBlock(segmentCounter, &segment[0], segmentSize);
                    segmentPos = 0;
                    segmentCounter++;
                }                
            }

            file.tableHeader.numberOfLines++;
        }
    }

    // write last incomplete segment to file
    if(segmentPos != 0) {
        file.addBlock(segmentCounter, &segment[0], segmentPos);
    }

    // update header in file for the final number of lines for the case,
    // that there were invalid lines
    if(file.updateHeader() == false) {
        return false;
    }

    // debug-output
    //file.print();

    return true;
}

