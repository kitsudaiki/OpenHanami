/**
 * @file        image_dataset_file.h
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

#ifndef HANAMI_IMAGEDATASETFILE_H
#define HANAMI_IMAGEDATASETFILE_H

#include <hanami_files/dataset_files/dataset_file.h>

class ImageDataSetFile : public DataSetFile
{
   public:
    struct ImageTypeHeader {
        uint64_t numberOfInputsX = 0;
        uint64_t numberOfInputsY = 0;
        uint64_t numberOfOutputs = 0;
        uint64_t numberOfImages = 0;
    };

    ImageDataSetFile(const std::string& filePath);
    ImageDataSetFile(Hanami::BinaryFile* file);
    ~ImageDataSetFile();
    bool updateHeader(Hanami::ErrorContainer& error);
    bool getPayload(Hanami::DataBuffer& result,
                    Hanami::ErrorContainer& error,
                    const std::string& columnName = "");

    ImageTypeHeader imageHeader;

   protected:
    void initHeader();
    void readHeader(const uint8_t* u8buffer);
};

#endif  // HANAMI_IMAGEDATASETFILE_H
