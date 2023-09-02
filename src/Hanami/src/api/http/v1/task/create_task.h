/**
 * @file        create_image_train_task.h
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

#ifndef HANAMI_CREATE_IMAGE_TRAINTASK_H
#define HANAMI_CREATE_IMAGE_TRAINTASK_H

#include <common.h>
#include <api/endpoint_processing/blossom.h>


class Cluster;

class CreateTask
        : public Blossom
{
public:
    CreateTask();

protected:
    bool runTask(BlossomIO &blossomIO,
                 const Hanami::DataMap &context,
                 BlossomStatus &status,
                 Hanami::ErrorContainer &error);

private:
    bool imageTask(std::string &taskUuid,
                   const std::string &name,
                   const std::string &taskType,
                   const UserContext &userContext,
                   Cluster* cluster,
                   Hanami::JsonItem &dataSetInfo,
                   BlossomStatus &status,
                   Hanami::ErrorContainer &error);

    bool tableTask(std::string &taskUuid,
                   const std::string &name,
                   const std::string &taskType,
                   const UserContext &userContext,
                   Cluster* cluster,
                   Hanami::JsonItem &dataSetInfo,
                   BlossomStatus &status,
                   Hanami::ErrorContainer &error);
};

#endif // HANAMI_CREATE_IMAGE_TRAINTASK_H
