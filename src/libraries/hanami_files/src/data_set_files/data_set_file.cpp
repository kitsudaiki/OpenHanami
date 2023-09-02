/**
 * @file        data_set_file.cpp
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

#include <hanami_files/data_set_files/data_set_file.h>

#include <hanami_common/files/binary_file.h>

#include <hanami_files/data_set_files/image_data_set_file.h>
#include <hanami_files/data_set_files/table_data_set_file.h>

/**
 * @brief constructor
 *
 * @param filePath path to file
 */
DataSetFile::DataSetFile(const std::string &filePath)
{
    m_targetFile = new Kitsunemimi::BinaryFile(filePath);
}

/**
 * @brief constructor
 *
 * @param file pointer to binary-file object
 */
DataSetFile::DataSetFile(Kitsunemimi::BinaryFile* file)
{
    m_targetFile = file;
}

/**
 * @brief destructor
 */
DataSetFile::~DataSetFile()
{
    delete m_targetFile;
}

/**
 * @brief initialize a new file with the already created header
 *
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
DataSetFile::initNewFile(Kitsunemimi::ErrorContainer &error)
{
    initHeader();

    // allocate storage
    if(m_targetFile->allocateStorage(m_totalFileSize, error) == false)
    {
        LOG_ERROR(error);
        // TODO: error-message
        return false;
    }

    // prepare dataset-header
    DataSetHeader dataSetHeader;
    dataSetHeader.type = type;
    uint32_t nameSize = name.size();
    if(nameSize > 255) {
        nameSize = 255;
    }
    memcpy(dataSetHeader.name, name.c_str(), nameSize);
    dataSetHeader.name[nameSize] = '\0';

    // write dataset-header to file
    if(m_targetFile->writeDataIntoFile(&dataSetHeader, 0, sizeof(DataSetHeader), error) == false)
    {
        error.addMeesage("Failed to write data-set to disc");
        return false;
    }

    // write data to file
    return updateHeader(error);
}

/**
 * @brief read complete file into memory
 *
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
DataSetFile::readFromFile(Kitsunemimi::ErrorContainer &error)
{
    // create complete file
    Kitsunemimi::DataBuffer buffer;
    if(m_targetFile->readCompleteFile(buffer, error) == false)
    {
        error.addMeesage("Faile to read data of data-set from disc");
        return false;
    }

    // prepare
    const uint8_t* u8buffer = static_cast<const uint8_t*>(buffer.data);

    // read data-set-header
    DataSetHeader dataSetHeader;
    memcpy(&dataSetHeader, u8buffer, sizeof(DataSetHeader));
    type = static_cast<DataSetType>(dataSetHeader.type);
    name = dataSetHeader.name;

    readHeader(u8buffer);

    return true;
}

/**
 * @brief add data to the file
 *
 * @param pos value-position, where to start to write to file
 * @param data values to write
 * @param numberOfValues number of values to write
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
DataSetFile::addBlock(const uint64_t pos,
                      const float* data,
                      const u_int64_t numberOfValues,
                      Kitsunemimi::ErrorContainer &error)
{
    // check size to not write over the end of the file
    if(m_headerSize + ((pos + numberOfValues) * sizeof(float)) > m_totalFileSize)
    {
        // TODO: error-message
        return false;
    }

    // add add data to file
    if(m_targetFile->writeDataIntoFile(data,
                                       m_headerSize + pos * sizeof(float),
                                       numberOfValues * sizeof(float),
                                       error) == false)
    {
        error.addMeesage("Failed to write block into data-set");
        return false;
    }

    return true;
}

/**
 * @brief read file as data-set
 *
 * @param filePath path to file
 * @param error reference for error-output
 *
 * @return pointer to file-handler, if successful, else nullptr
 */
DataSetFile*
readDataSetFile(const std::string &filePath,
                Kitsunemimi::ErrorContainer &error)
{
    // read header of file to identify type
    Kitsunemimi::BinaryFile* targetFile = new Kitsunemimi::BinaryFile(filePath);
    DataSetFile::DataSetHeader header;
    if(targetFile->readDataFromFile(&header, 0 , sizeof(DataSetFile::DataSetHeader), error) == false)
    {
        error.addMeesage("failed to read dataset-file");
        return nullptr;
    }

    // create file-handling object based on the type from the header
    DataSetFile* file = nullptr;
    if(header.type == DataSetFile::IMAGE_TYPE) {
        file = new ImageDataSetFile(targetFile);
    }
    else if(header.type == DataSetFile::TABLE_TYPE) {
        file = new TableDataSetFile(targetFile);
    }

    if(file != nullptr
            && file->readFromFile(error) == false)
    {
        delete file;
        return nullptr;
    }

    return file;
}
