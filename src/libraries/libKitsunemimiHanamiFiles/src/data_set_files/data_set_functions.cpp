/**
 * @file        data_set_functions.cpp
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

#include <libKitsunemimiHanamiFiles/data_set_files/data_set_functions.h>

#include <libKitsunemimiHanamiFiles/data_set_files/data_set_file.h>
#include <libKitsunemimiHanamiFiles/data_set_files/image_data_set_file.h>
#include <libKitsunemimiHanamiFiles/data_set_files/table_data_set_file.h>

#include <libKitsunemimiJson/json_item.h>
#include <libKitsunemimiCommon/files/binary_file.h>

/**
 * @brief getDataSetPayload
 * @param location
 * @param error
 * @param columnName
 * @return
 */
bool
getDataSetPayload(Kitsunemimi::DataBuffer &result,
                  const std::string &location,
                  Kitsunemimi::ErrorContainer &error,
                  const std::string &columnName)
{
    // init file
    DataSetFile* file = readDataSetFile(location, error);
    if(file == nullptr) {
        return false;
    }

    // get payload
    if(file->getPayload(result, error, columnName) == false)
    {
        delete file;
        error.addMeesage("Failed to get payload.");
        return false;
    }
    delete file;

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
getHeaderInformation(Kitsunemimi::JsonItem &result,
                     const std::string &location,
                     Kitsunemimi::ErrorContainer &error)
{
    bool ret = false;

    DataSetFile* file = readDataSetFile(location, error);
    if(file == nullptr) {
        return ret;
    }

    do
    {
        // read data-set-header
        if(file->readFromFile(error) == false)
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
            for(const TableDataSetFile::TableHeaderEntry &entry : imgT->tableColumns)
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
    }
    while(false);

    delete file;

    return ret;
}
