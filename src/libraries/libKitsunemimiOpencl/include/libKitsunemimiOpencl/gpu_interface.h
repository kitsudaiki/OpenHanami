/**
 * @file        gpu_interface.h
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

#ifndef OPENCL_H
#define OPENCL_H

#include <iostream>
#include <vector>
#include <map>
#include <string>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>

#include <libKitsunemimiOpencl/gpu_data.h>
#include <libKitsunemimiCommon/logger.h>

namespace Kitsunemimi
{

class GpuInterface
{
public:
    GpuInterface(const cl::Device &device);
    ~GpuInterface();

    // initializing
    bool initCopyToDevice(GpuData &data,
                          ErrorContainer &error);

    bool addKernel(GpuData &data,
                   const std::string &kernelName,
                   const std::string &kernelCode,
                   ErrorContainer &error);
    bool bindKernelToBuffer(GpuData &data,
                            const std::string &kernelName,
                            const std::string &bufferName,
                            ErrorContainer &error);
    bool setLocalMemory(GpuData &data,
                        const std::string &kernelName,
                        const uint32_t localMemorySize,
                        ErrorContainer &error);

    bool closeDevice(GpuData &data);

    // runtime
    bool updateBufferOnDevice(GpuData &data,
                              const std::string &bufferName,
                              ErrorContainer &error,
                              uint64_t numberOfObjects = 0,
                              const uint64_t offset = 0);
    bool run(GpuData &data,
             const std::string &kernelName,
             ErrorContainer &error);
    bool copyFromDevice(GpuData &data,
                        const std::string &bufferName,
                        ErrorContainer &error);

    // common getter
    const std::string getDeviceName();

    // getter for memory information
    uint64_t getLocalMemorySize();
    uint64_t getGlobalMemorySize();
    uint64_t getMaxMemAllocSize();

    // getter for work-group information
    uint64_t getMaxWorkGroupSize();
    const WorkerDim getMaxWorkItemSize();
    uint64_t getMaxWorkItemDimension();

    cl::Device m_device;
    cl::Context m_context;
    cl::CommandQueue m_queue;

private:
    bool validateWorkerGroupSize(const GpuData &data,
                                 ErrorContainer &error);
};

}

#endif // OPENCL_H
