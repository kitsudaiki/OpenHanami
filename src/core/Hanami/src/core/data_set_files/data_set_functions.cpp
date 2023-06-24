#include "data_set_functions.h"

#include <hanami_root.h>
#include <core/data_set_files/data_set_file.h>
#include <core/data_set_files/image_data_set_file.h>
#include <core/data_set_files/table_data_set_file.h>
#include <database/data_set_table.h>




#include <libKitsunemimiJson/json_item.h>
#include <libKitsunemimiCommon/files/binary_file.h>

using namespace Kitsunemimi::Hanami;

float*
getDataSetPayload(const std::string &location,
                  Kitsunemimi::ErrorContainer &error,
                  const std::string &columnName)
{
    // init file
    DataSetFile* file = readDataSetFile(location);
    if(file == nullptr) {
        return nullptr;
    }

    // get payload
    uint64_t payloadSize = 0;
    float* payload = file->getPayload(payloadSize, columnName);
    if(payload == nullptr)
    {
        // TODO: error
        return nullptr;
    }

    return payload;
}

/**
 * @brief getDateSetInfo
 * @param dataUuid
 * @param error
 * @return
 */
bool
getDateSetInfo(Kitsunemimi::JsonItem &result,
               const std::string &dataUuid,
               const Kitsunemimi::DataMap &context,
               Kitsunemimi::ErrorContainer &error)
{
    const UserContext userContext(context);

    if(HanamiRoot::dataSetTable->getDataSet(result,
                                            dataUuid,
                                            userContext,
                                            error,
                                            true) == false)
    {
        return false;
    }

    // get file information
    const std::string location = result.get("location").getString();
    if(getHeaderInformation(result, location, error) == false)
    {
        error.addMeesage("Failed the read information from file '" + location + "'");
        return false;
    }

    // remove irrelevant fields
    result.remove("temp_files");

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
