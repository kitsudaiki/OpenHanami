/**
 * @file        create_train_task_v1_0.h
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

#include <api/endpoint_processing/blossom.h>
#include <core/cluster/task.h>

class Cluster;

class CreateTrainTaskV1M0 : public Blossom
{
   public:
    CreateTrainTaskV1M0();

   protected:
    bool runTask(BlossomIO& blossomIO,
                 const Hanami::UserContext& userContext,
                 BlossomStatus& status,
                 Hanami::ErrorContainer& error);

   private:
    ReturnStatus fillTaskIo(DataSetFileHandle& taskIo,
                            const Hanami::UserContext& userContext,
                            const std::string& columnName,
                            const std::string& settings,
                            BlossomStatus& status,
                            Hanami::ErrorContainer& error);
};

#endif  // HANAMI_CREATE_IMAGE_TRAINTASK_H
