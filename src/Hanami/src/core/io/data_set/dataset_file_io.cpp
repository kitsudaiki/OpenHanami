/**
 * @file        dataset_file_io.cpp
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

#include "dataset_file_io.h"

/**
 * @brief open a data-set file
 *
 * @param result resulting handle to the open file
 * @param filePath path on the local disc to the data-set, which should be opend
 * @param error reference for error-output
 *
 * @return OK, INVALID_INPUT or ERROR
 */
ReturnStatus
openDataSetFile(DataSetFileHandle& result,
                const std::string& filePath,
                Hanami::ErrorContainer& error)
{
    // check source
    if (std::filesystem::exists(filePath) == false) {
        error.addMessage("Data-set file '" + filePath + "' doesn't exist.");
        return INVALID_INPUT;
    }

    // open file
    result.targetFile = new Hanami::BinaryFile(filePath);
    if (result.targetFile->isOpen() == false) {
        error.addMessage("Failed to open data-set file '" + filePath + "'.");
        return ERROR;
    }

    // check for minimum size
    if (result.targetFile->fileSize < sizeof(DataSetHeader)) {
        error.addMessage("Data-set file '" + filePath + "' is too small for the header.");
        return INVALID_INPUT;
    }

    // read header of file to identify type
    if (result.targetFile->readDataFromFile(&result.header, 0, sizeof(DataSetHeader), error)
        == false)
    {
        error.addMessage("Failed to read header of data-set-file");
        return ERROR;
    }

    return OK;
}

/**
 * @brief create a new data-set file
 *
 * @param result handle to the new created file
 * @param filePath path on the local disc, where the file should be created
 * @param name name of the new data-set
 * @param type tpe of the new data-set
 * @param numberOfColumns maximum number of columns, which should be stored by the data-set
 * @param error reference for error-output
 *
 * @return OK, INVALID_INPUT or ERROR
 */
ReturnStatus
initNewDataSetFile(DataSetFileHandle& result,
                   const std::string& filePath,
                   const std::string& name,
                   const DataSetType type,
                   const uint64_t numberOfColumns,
                   Hanami::ErrorContainer& error)
{
    // check source
    if (std::filesystem::exists(filePath) == true) {
        error.addMessage("Data-set file '" + filePath + "' already exist.");
        return INVALID_INPUT;
    }

    // open file
    result.targetFile = new Hanami::BinaryFile(filePath);
    if (result.targetFile->isOpen() == false) {
        error.addMessage("Failed to open data-set file '" + filePath + "'.");
        return ERROR;
    }

    // allocate initaial storage for the header
    if (result.targetFile->allocateStorage(sizeof(DataSetHeader), error) == false) {
        error.addMessage("Failed to allocate storage in file '" + filePath
                         + "' for the data-set header");
        return ERROR;
    }

    // set name in header
    if (result.header.setName(name) == false) {
        error.addMessage("New data-set name '" + name + "' is invalid");
        return INVALID_INPUT;
    }

    // write header to target
    result.header.dataType = type;
    result.header.numberOfColumns = numberOfColumns;
    if (result.updateHeaderInFile(error) == false) {
        error.addMessage("Failed to update data-set header in file");
        return ERROR;
    }
    result.header.fileSize = result.targetFile->fileSize;

    return OK;
}

/**
 * @brief get a section of data from the data-set
 *
 * @param result reference to list with all read values
 * @param fileHandle handle to the file and header
 * @param selector object to select a section within the dataset
 * @param error reference for error-output
 *
 * @return OK, INVALID_INPUT or ERROR
 */
ReturnStatus
getDataFromDataSet(std::vector<float>& result,
                   const DataSetFileHandle& fileHandle,
                   const DataSetSelector selector,
                   Hanami::ErrorContainer& error)
{
    if (fileHandle.targetFile->isOpen() == false) {
        error.addMessage("Data-set file '" + fileHandle.targetFile->filePath + "' is not open");
        return INVALID_INPUT;
    }

    if (fileHandle.header.numberOfColumns == 0) {
        return INVALID_INPUT;
    }

    // check ranges
    if (selector.endColumn > fileHandle.header.numberOfColumns
        || selector.startColumn > fileHandle.header.numberOfColumns)
    {
        return INVALID_INPUT;
    }
    if (selector.endRow > fileHandle.header.numberOfRows
        || selector.startRow > fileHandle.header.numberOfRows)
    {
        return INVALID_INPUT;
    }

    const uint64_t numberOfValues
        = (selector.endRow - selector.startRow) * (selector.endColumn - selector.startColumn);
    result.clear();
    // TODO: check if resize successful
    result.resize(numberOfValues);

    // read block from file into buffer
    uint64_t numberOfBytes
        = (selector.endRow - selector.startRow) * fileHandle.header.numberOfColumns;
    uint64_t byteOffset = selector.startRow * fileHandle.header.numberOfColumns;
    switch (fileHandle.header.dataType) {
        case UNDEFINED_TYPE:
            error.addMessage("Invalid data-type defined in file '" + fileHandle.targetFile->filePath
                             + "'");
            return ERROR;
        case UNIN8_TYPE:
            break;
        case UNIN32_TYPE:
            numberOfBytes *= sizeof(uint32_t);
            byteOffset *= sizeof(uint32_t);
            break;
        case UNIN64_TYPE:
            numberOfBytes *= sizeof(uint64_t);
            byteOffset *= sizeof(uint64_t);
            break;
        case FLOAT_TYPE:
            numberOfBytes *= sizeof(float);
            byteOffset *= sizeof(float);
            break;
    }
    byteOffset += sizeof(DataSetHeader);

    // prepare temp-buffer
    Hanami::DataBuffer buffer;
    if (Hanami::allocateBlocks_DataBuffer(buffer, Hanami::calcBytesToBlocks(numberOfBytes))
        == false)
    {
        error.addMessage("Failed to allocate buffer to read from dataset-file '"
                         + fileHandle.targetFile->filePath + "'");
        return ERROR;
    }

    if (fileHandle.targetFile->readDataFromFile(buffer.data, byteOffset, numberOfBytes, error)
        == false)
    {
        error.addMessage("Failed to read data of dataset-file '" + fileHandle.targetFile->filePath
                         + "'");
        return ERROR;
    }

    switch (fileHandle.header.dataType) {
        case UNDEFINED_TYPE:
            error.addMessage("Invalid data-type defined in file '" + fileHandle.targetFile->filePath
                             + "'");
            return ERROR;
        case UNIN8_TYPE:
            copyDataSetDate<uint8_t>(result, buffer, selector, fileHandle.header.numberOfColumns);
            break;
        case UNIN32_TYPE:
            copyDataSetDate<uint32_t>(result, buffer, selector, fileHandle.header.numberOfColumns);
            break;
        case UNIN64_TYPE:
            copyDataSetDate<uint64_t>(result, buffer, selector, fileHandle.header.numberOfColumns);
            break;
        case FLOAT_TYPE:
            copyDataSetDate<float>(result, buffer, selector, fileHandle.header.numberOfColumns);
            break;
    }

    return OK;
}
