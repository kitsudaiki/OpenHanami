/**
 * @file        image_dataset_file.cpp
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

#include <hanami_common/files/binary_file.h>
#include <hanami_files/dataset_files/image_dataset_file.h>

/**
 * @brief constructor
 *
 * @param filePath path to file
 */
ImageDataSetFile::ImageDataSetFile(const std::string& filePath) : DataSetFile(filePath) {}

/**
 * @brief constructor
 *
 * @param file pointer to binary-file object
 */
ImageDataSetFile::ImageDataSetFile(Hanami::BinaryFile* file) : DataSetFile(file) {}

/**
 * @brief destructor
 */
ImageDataSetFile::~ImageDataSetFile() {}

/**
 * @brief init header-sizes
 */
void
ImageDataSetFile::initHeader()
{
    m_headerSize = sizeof(DataSetHeader) + sizeof(ImageTypeHeader);

    uint64_t lineSize
        = (imageHeader.numberOfInputsX * imageHeader.numberOfInputsY) + imageHeader.numberOfOutputs;
    m_totalFileSize = m_headerSize + (lineSize * imageHeader.numberOfImages * sizeof(float));
}

/**
 * @brief read header from buffer
 *
 * @param u8buffer buffer to read
 */
void
ImageDataSetFile::readHeader(const uint8_t* u8buffer)
{
    // read image-header
    m_headerSize = sizeof(DataSetHeader) + sizeof(ImageTypeHeader);
    memcpy(&imageHeader, &u8buffer[sizeof(DataSetHeader)], sizeof(ImageTypeHeader));

    // get sizes
    uint64_t lineSize
        = (imageHeader.numberOfInputsX * imageHeader.numberOfInputsY) + imageHeader.numberOfOutputs;
    m_totalFileSize = m_headerSize + (lineSize * imageHeader.numberOfImages * sizeof(float));
}

/**
 * @brief update header in file
 *
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
ImageDataSetFile::updateHeader(Hanami::ErrorContainer& error)
{
    // write image-header to file
    if (m_targetFile->writeDataIntoFile(
            &imageHeader, sizeof(DataSetHeader), sizeof(ImageTypeHeader), error)
        == false)
    {
        error.addMessage("Failed to update header of image-file");
        return false;
    }

    return true;
}

/**
 * @brief read payload of the image-file
 *
 * @param result reference to buffer for the read data
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
ImageDataSetFile::getPayload(Hanami::DataBuffer& result,
                             Hanami::ErrorContainer& error,
                             const std::string&)
{
    const uint64_t payloadSize = m_totalFileSize - m_headerSize;
    Hanami::allocateBlocks_DataBuffer(result, Hanami::calcBytesToBlocks(payloadSize));
    if (m_targetFile->readDataFromFile(result.data, m_headerSize, payloadSize, error) == false) {
        error.addMessage("Failed to read data of image-dataset-file");
        return false;
    }

    return true;
}
