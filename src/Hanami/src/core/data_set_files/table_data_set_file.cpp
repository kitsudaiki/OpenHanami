/**
 * @file        table_data_set_file.cpp
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

#include "table_data_set_file.h"

#include <libKitsunemimiCommon/files/binary_file.h>

/**
 * @brief constructor
 *
 * @param filePath path to file
 */
TableDataSetFile::TableDataSetFile(const std::string &filePath)
    : DataSetFile(filePath) {}

/**
 * @brief constructor
 *
 * @param file pointer to binary-file object
 */
TableDataSetFile::TableDataSetFile(Kitsunemimi::BinaryFile* file)
    : DataSetFile(file) {}

/**
 * @brief destructor
 */
TableDataSetFile::~TableDataSetFile() {}

/**
 * @brief init header-sizes
 */
void
TableDataSetFile::initHeader()
{
    m_headerSize = sizeof(DataSetHeader) + sizeof(TableTypeHeader);
    m_headerSize += tableColumns.size() * sizeof(TableHeaderEntry);

    tableHeader.numberOfColumns = tableColumns.size();
    m_totalFileSize = m_headerSize;
    m_totalFileSize += tableHeader.numberOfColumns * sizeof(float) * tableHeader.numberOfLines;
}

/**
 * @brief read header from buffer
 *
 * @param u8buffer buffer to read
 */
void
TableDataSetFile::readHeader(const uint8_t* u8buffer)
{
    // read table-header
    m_headerSize = sizeof(DataSetHeader) + sizeof(TableTypeHeader);
    memcpy(&tableHeader, &u8buffer[sizeof(DataSetHeader)], sizeof(TableTypeHeader));

    // header header
    for(uint64_t i = 0; i < tableHeader.numberOfColumns; i++)
    {
        TableHeaderEntry entry;
        memcpy(&entry,
               &u8buffer[m_headerSize + (i * sizeof(TableHeaderEntry))],
               sizeof(TableHeaderEntry));
        tableColumns.push_back(entry);
    }

    // get sizes
    m_headerSize += tableHeader.numberOfColumns * sizeof(TableHeaderEntry);
    m_totalFileSize = m_headerSize;
    m_totalFileSize += tableHeader.numberOfColumns * sizeof(float) * tableHeader.numberOfLines;
}

/**
 * @brief update header in file
 *
 * @return true, if successful, else false
 */
bool
TableDataSetFile::updateHeader()
{
    // write table-header to file
    Kitsunemimi::ErrorContainer error;
    if(m_targetFile->writeDataIntoFile(&tableHeader,
                                       sizeof(DataSetHeader),
                                       sizeof(TableTypeHeader),
                                       error) == false)
    {
        LOG_ERROR(error);
        return false;
    }

    // write table-header-entries to file
    const uint64_t offset = sizeof(DataSetHeader) + sizeof(TableTypeHeader);
    for(uint64_t i = 0; i < tableColumns.size(); i++)
    {
        if(m_targetFile->writeDataIntoFile(&tableColumns[i],
                                           offset + (i * sizeof(TableHeaderEntry)),
                                           sizeof(TableHeaderEntry),
                                           error) == false)
        {
            LOG_ERROR(error);
            return false;
        }
    }

    return true;
}

/**
 * @brief get pointer to payload of a file
 *
 * @param payloadSize reference for size of the read payload
 *
 * @return pointer to the payload
 */
float*
TableDataSetFile::getPayload(uint64_t &payloadSize,
                             const std::string &columnName)
{
    Kitsunemimi::ErrorContainer error;

    float* payload = new float[(m_totalFileSize - m_headerSize) / sizeof(float)];
    if(m_targetFile->readDataFromFile(payload,
                                      m_headerSize,
                                      m_totalFileSize - m_headerSize,
                                      error) == false)
    {
        //TODO: handle error
        LOG_ERROR(error);
        return payload;
    }

    uint64_t columnPos = 0;
    for(uint64_t i = 0; i < tableColumns.size(); i++)
    {
        if(tableColumns[i].name == columnName) {
            columnPos = i;
        }
    }

    payloadSize = tableHeader.numberOfLines * sizeof(float);
    float* filteredData = new float[tableHeader.numberOfLines];
    for(uint64_t line = 0; line < tableHeader.numberOfLines; line++) {
        filteredData[line] = payload[line * tableHeader.numberOfColumns + columnPos];
    }

    delete[] payload;

    return filteredData;
}

/**
 * @brief print-function for manually debugging only
 */
void
TableDataSetFile::print()
{
    std::cout<<"======================================================="<<std::endl;
    std::cout<<"====================    PRINT    ======================"<<std::endl;
    std::cout<<"======================================================="<<std::endl;
    std::cout<<std::endl;

    Kitsunemimi::ErrorContainer error;

    Kitsunemimi::DataBuffer completeFile;
    if(m_targetFile->readCompleteFile(completeFile, error) == false)
    {
        error.addMeesage("Failed to read file");
        LOG_ERROR(error);
    }

    // read data-set-header
    DataSetHeader dataSetHeader;
    memcpy(&dataSetHeader, completeFile.data, sizeof(DataSetHeader));
    std::cout<<"name: "<<dataSetHeader.name<<std::endl;

    // read table-header
    const uint8_t* u8buffer = static_cast<uint8_t*>(completeFile.data);
    uint32_t headerSize = sizeof(DataSetHeader) + sizeof(TableTypeHeader);
    memcpy(&tableHeader, &u8buffer[sizeof(DataSetHeader)], sizeof(TableTypeHeader));

    std::cout<<"number of columns: "<<tableHeader.numberOfColumns<<std::endl;
    std::cout<<"number of lines: "<<tableHeader.numberOfLines<<std::endl;
    std::cout<<std::endl;

    // header header
    for(uint64_t i = 0; i < tableHeader.numberOfColumns; i++)
    {
        std::cout<<"column:"<<std::endl;
        TableHeaderEntry entry;
        memcpy(&entry,
               &u8buffer[headerSize + (i * sizeof(TableHeaderEntry))],
               sizeof(TableHeaderEntry));
        std::cout<<"    name: "<<entry.name<<std::endl;
        std::cout<<"    avg: "<<entry.averageVal<<std::endl;
        std::cout<<"    max: "<<entry.maxVal<<std::endl;
        std::cout<<"    multi: "<<entry.multiplicator<<std::endl;
        std::cout<<std::endl;
    }
    std::cout<<std::endl;

    // get sizes
    headerSize += tableHeader.numberOfColumns * sizeof(TableHeaderEntry);


    std::cout<<"content:"<<std::endl;
    const float* fbuffer = reinterpret_cast<const float*>(&u8buffer[headerSize]);

    for(uint64_t line = 0; line < tableHeader.numberOfLines; line++)
    {
        for(uint64_t col = 0; col < tableHeader.numberOfColumns; col++)
        {
            std::cout<<fbuffer[line * tableHeader.numberOfColumns + col];
            std::cout<<"   ";
        }
        std::cout<<"\n";
    }

    std::cout<<"======================================================="<<std::endl;
}
