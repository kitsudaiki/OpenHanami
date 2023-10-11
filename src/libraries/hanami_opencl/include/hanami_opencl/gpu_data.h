/**
 * @file        gpu_data.h
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

#ifndef GPU_DATA_H
#define GPU_DATA_H

#include <hanami_common/buffer/data_buffer.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>

namespace Hanami
{
class GpuInterface;

struct WorkerDim {
    uint64_t x = 1;
    uint64_t y = 1;
    uint64_t z = 1;
};

class GpuData
{
   public:
    WorkerDim numberOfWg;
    WorkerDim threadsPerWg;

    GpuData();

    bool addBuffer(const std::string &name,
                   const uint64_t numberOfObjects,
                   const uint64_t objectSize,
                   const bool useHostPtr = false,
                   void *data = nullptr);
    bool addValue(const std::string &name, const uint64_t value);

    bool containsBuffer(const std::string &name);
    void *getBufferData(const std::string &name);

   private:
    friend GpuInterface;

    struct WorkerBuffer {
        bool isValue = false;
        uint64_t value = 0;
        void *data = nullptr;
        uint64_t numberOfBytes = 0;
        uint64_t numberOfObjects = 0;
        bool useHostPtr = false;
        bool allowBufferDeleteAfterClose = true;
        cl::Buffer clBuffer;
    };

    struct KernelDef {
        std::string id = "";
        std::string kernelCode = "";
        cl::Kernel kernel;
        std::map<std::string, uint32_t> arguments;
        uint32_t localBufferSize = 0;
        uint32_t argumentCounter = 0;
    };

    std::map<std::string, WorkerBuffer> m_buffer;
    std::map<std::string, KernelDef> m_kernel;

    WorkerBuffer *getBuffer(const std::string &name);

    bool containsKernel(const std::string &name);
    KernelDef *getKernel(const std::string &name);

    uint32_t getArgPosition(KernelDef *kernelDef, const std::string &bufferName);
};

}  // namespace Hanami

#endif  // GPU_DATA_H
