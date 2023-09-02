/**
 * @file        data_set.h
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

#ifndef KITSUNEMIMI_HANAMISDK_DATA_SET_H
#define KITSUNEMIMI_HANAMISDK_DATA_SET_H

#include <hanami_common/logger.h>

namespace HanamiAI
{

bool uploadCsvData(std::string &result,
                   const std::string &dataSetName,
                   const std::string &inputFilePath,
                   Hanami::ErrorContainer &error);

bool uploadMnistData(std::string &result,
                     const std::string &dataSetName,
                     const std::string &inputFilePath,
                     const std::string &labelFilePath,
                     Hanami::ErrorContainer &error);

bool checkDataset(std::string &result,
                  const std::string &dataUuid,
                  const std::string &resultUuid,
                  Hanami::ErrorContainer &error);

bool getDataset(std::string &result,
                const std::string &dataUuid,
                Hanami::ErrorContainer &error);

bool listDatasets(std::string &result,
                  Hanami::ErrorContainer &error);

bool deleteDataset(std::string &result,
                   const std::string &dataUuid,
                   Hanami::ErrorContainer &error);

bool getDatasetProgress(std::string &result,
                   const std::string &dataUuid,
                   Hanami::ErrorContainer &error);

} // namespace HanamiAI

#endif // KITSUNEMIMI_HANAMISDK_DATA_SET_H
