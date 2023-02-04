/**
 * @file        get_data_set.cpp
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

#include "get_data_set.h"

#include <shiori_root.h>
#include <database/data_set_table.h>
#include <core/data_set_files/data_set_file.h>
#include <core/data_set_files/image_data_set_file.h>
#include <core/data_set_files/table_data_set_file.h>

#include <libKitsunemimiHanamiCommon/enums.h>
#include <libKitsunemimiHanamiCommon/defines.h>

#include <libKitsunemimiCrypto/common.h>
#include <libKitsunemimiJson/json_item.h>

using namespace Kitsunemimi::Hanami;

GetDataSet::GetDataSet()
    : Blossom("Get information of a specific data-set.")
{
    //----------------------------------------------------------------------------------------------
    // input
    //----------------------------------------------------------------------------------------------

    registerInputField("uuid",
                       SAKURA_STRING_TYPE,
                       true,
                       "UUID of the data-set set to delete.");
    assert(addFieldRegex("uuid", UUID_REGEX));

    //----------------------------------------------------------------------------------------------
    // output
    //----------------------------------------------------------------------------------------------

    registerOutputField("uuid",
                        SAKURA_STRING_TYPE,
                        "UUID of the data-set.");
    registerOutputField("name",
                        SAKURA_STRING_TYPE,
                        "Name of the data-set.");
    registerOutputField("owner_id",
                        SAKURA_STRING_TYPE,
                        "ID of the user, who created the data-set.");
    registerOutputField("project_id",
                        SAKURA_STRING_TYPE,
                        "ID of the project, where the data-set belongs to.");
    registerOutputField("visibility",
                        SAKURA_STRING_TYPE,
                        "Visibility of the data-set (private, shared, public).");
    registerOutputField("location",
                        SAKURA_STRING_TYPE,
                        "Local file-path of the data-set.");
    registerOutputField("type",
                        SAKURA_STRING_TYPE,
                        "Type of the new set (csv or mnist)");
    registerOutputField("inputs",
                        SAKURA_INT_TYPE,
                        "Number of inputs.");
    registerOutputField("outputs",
                        SAKURA_INT_TYPE,
                        "Number of outputs.");
    registerOutputField("lines",
                        SAKURA_INT_TYPE,
                        "Number of lines.");

    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------
}

/**
 * @brief runTask
 */
bool
GetDataSet::runTask(BlossomIO &blossomIO,
                      const Kitsunemimi::DataMap &context,
                      BlossomStatus &status,
                      Kitsunemimi::ErrorContainer &error)
{
    const std::string dataUuid = blossomIO.input.get("uuid").getString();
    const Kitsunemimi::Hanami::UserContext userContext(context);

    if(ShioriRoot::dataSetTable->getDataSet(blossomIO.output,
                                            dataUuid,
                                            userContext,
                                            error,
                                            true) == false)
    {
        status.statusCode = Kitsunemimi::Hanami::INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // get file information
    const std::string location = blossomIO.output.get("location").getString();
    if(getHeaderInformation(blossomIO.output, location, error) == false)
    {
        error.addMeesage("Failed the read information from file '" + location + "'");
        status.statusCode = Kitsunemimi::Hanami::INTERNAL_SERVER_ERROR_RTYPE;
        return false;
    }

    // remove irrelevant fields
    blossomIO.output.remove("temp_files");

    return true;
}

/**
 * @brief get information from header of file
 *
 * @param result reference for result-output
 * @param location location of the file to read
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
GetDataSet::getHeaderInformation(Kitsunemimi::JsonItem &result,
                                 const std::string &location,
                                 Kitsunemimi::ErrorContainer &error)
{
    bool ret = false;

    DataSetFile* file = readDataSetFile(location);
    if(file == nullptr) {
        return ret;
    }

    do
    {
        // read data-set-header
        if(file->readFromFile() == false)
        {
            error.addMeesage("Failed to read header from file '" + location + "'");
            break;
        }

        if(file->type == DataSetFile::IMAGE_TYPE)
        {
            ImageDataSetFile* imgF = dynamic_cast<ImageDataSetFile*>(file);
            if(imgF == nullptr) {
                break;
            }

            // write information to result
            const uint64_t size = imgF->imageHeader.numberOfInputsX * imgF->imageHeader.numberOfInputsY;
            result.insert("inputs", static_cast<long>(size));
            result.insert("outputs", static_cast<long>(imgF->imageHeader.numberOfOutputs));
            result.insert("lines", static_cast<long>(imgF->imageHeader.numberOfImages));
            // result.insert("average_value", static_cast<float>(imgF->imageHeader.avgValue));
            // result.insert("max_value", static_cast<float>(imgF->imageHeader.maxValue));

            ret = true;
            break;
        }
        else if(file->type == DataSetFile::TABLE_TYPE)
        {
            TableDataSetFile* imgT = dynamic_cast<TableDataSetFile*>(file);
            if(imgT == nullptr) {
                break;
            }

            long inputs = 0;
            long outputs = 0;

            // get number of inputs and outputs
            for(const DataSetFile::TableHeaderEntry &entry : imgT->tableColumns)
            {
                if(entry.isInput) {
                    inputs++;
                }
                if(entry.isOutput) {
                    outputs++;
                }
            }

            result.insert("inputs", inputs);
            result.insert("outputs", outputs);
            result.insert("lines", static_cast<long>(imgT->tableHeader.numberOfLines));
            // result.insert("average_value", 0.0f);
            // result.insert("max_value", 0.0f);

            ret = true;
            break;
        }

        // TODO: handle other types
        break;
    }
    while(true);

    delete file;

    return ret;
}
