/**
 * @file        image_data_set_file.h
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

#ifndef SHIORIARCHIVE_IMAGEDATASETFILE_H
#define SHIORIARCHIVE_IMAGEDATASETFILE_H

#include <core/data_set_files/data_set_file.h>

class ImageDataSetFile
        : public DataSetFile
{
public:
    ImageDataSetFile(const std::string &filePath);
    ImageDataSetFile(Kitsunemimi::BinaryFile* file);
    ~ImageDataSetFile();
    bool updateHeader();
    float* getPayload(uint64_t &payloadSize,
                      const std::string &columnName = "");

    ImageTypeHeader imageHeader;

protected:
    void initHeader();
    void readHeader(const uint8_t* u8buffer);
};

#endif // SHIORIARCHIVE_IMAGEDATASETFILE_H
