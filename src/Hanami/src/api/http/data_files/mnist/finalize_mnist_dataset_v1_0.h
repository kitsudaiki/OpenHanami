/**
 * @file        finalize_mnist_dataset_v1_0.h
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

#ifndef HANAMI_MNIST_FINALIZE_DATA_SET_H
#define HANAMI_MNIST_FINALIZE_DATA_SET_H

#include <api/endpoint_processing/blossom.h>
#include <hanami_common/buffer/data_buffer.h>

class FinalizeMnistDataSetV1M0 : public Blossom
{
   public:
    FinalizeMnistDataSetV1M0();

   protected:
    bool runTask(BlossomIO& blossomIO,
                 const json& context,
                 BlossomStatus& status,
                 Hanami::ErrorContainer& error);

   private:
    bool convertMnistData(const std::string& filePath,
                          const std::string& name,
                          const Hanami::DataBuffer& inputBuffer,
                          const Hanami::DataBuffer& labelBuffer,
                          Hanami::ErrorContainer& error);
};

#endif  // HANAMI_MNIST_FINALIZE_DATA_SET_H
