/**
 * @file        structs.h
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

#ifndef KITSUNEMIMI_HANAMI_FILES_STRUCTS_H
#define KITSUNEMIMI_HANAMI_FILES_STRUCTS_H

#include <libKitsunemimiCommon/items/data_items.h>

namespace Kitsunemimi::Hanami
{

enum FileType
{
    UNDEFINED_FILE_TYPE = 0,
    IMAGE_FILE_TYPE = 1,
    TABLE_FILE_TYPE = 2,
    SNAPSHOT_FILE_TYPE = 3,
};

struct FileHeader
{
    uint8_t type = UNDEFINED_FILE_TYPE;
    char name[255];

    void setName(const std::string &name)
    {
        const uint32_t nameSize = (name.size() <= 254) ? name.size() : 254;
        memcpy(this->name, name.c_str(), nameSize);
        this->name[nameSize] = '\0';
    }
};

struct ImageTypeHeader
{
    uint64_t numberOfInputsX = 0;
    uint64_t numberOfInputsY = 0;
    uint64_t numberOfOutputs = 0;
    uint64_t numberOfImages = 0;
    float maxValue = 0.0f;
    float avgValue = 0.0f;
};

struct TableTypeHeader
{
    uint64_t numberOfColumns = 0;
    uint64_t numberOfLines = 0;
};

struct TableHeaderEntry
{
    char name[255];
    uint8_t padding1[1];
    float multiplicator = 1.0f;
    float averageVal = 0.0f;
    float maxVal = 0.0f;

    void setName(const std::string &name)
    {
        const uint32_t nameSize = (name.size() <= 254) ? name.size() : 254;
        memcpy(this->name, name.c_str(), nameSize);
        this->name[nameSize] = '\0';
    }
};

}

#endif // KITSUNEMIMI_HANAMI_FILES_STRUCTS_H
