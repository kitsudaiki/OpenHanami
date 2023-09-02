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

#ifndef HANAMI_TABLEDATASETFILE_H
#define HANAMI_TABLEDATASETFILE_H

#include <hanami_files/data_set_files/data_set_file.h>

class TableDataSetFile
        : public DataSetFile
{
public:
    struct TableTypeHeader
    {
        uint64_t numberOfColumns = 0;
        uint64_t numberOfLines = 0;
    };

    struct TableHeaderEntry
    {
        char name[256];
        bool isInput = false;
        bool isOutput = false;
        float multiplicator = 1.0f;
        float averageVal = 0.0f;
        float maxVal = 0.0f;

        void setName(const std::string &name)
        {
            uint32_t nameSize = name.size();
            if(nameSize > 255) {
                nameSize = 255;
            }
            memcpy(this->name, name.c_str(), nameSize);
            this->name[nameSize] = '\0';
        }
    };

    TableDataSetFile(const std::string &filePath);
    TableDataSetFile(Kitsunemimi::BinaryFile* file);
    ~TableDataSetFile();
    bool updateHeader(Kitsunemimi::ErrorContainer &error);
    bool getPayload(Kitsunemimi::DataBuffer &result,
                    Kitsunemimi::ErrorContainer &error,
                    const std::string &columnName = "");

    void print();

    TableTypeHeader tableHeader;
    std::vector<TableHeaderEntry> tableColumns;

protected:
    void initHeader();
    void readHeader(const uint8_t* u8buffer);
};

#endif // HANAMI_TABLEDATASETFILE_H
