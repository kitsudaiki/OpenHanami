/**
 * @file        gpu_data.cpp
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

#include <libKitsunemimiOpencl/gpu_data.h>

namespace Kitsunemimi
{

GpuData::GpuData() {}

/**
 * @brief register new buffer
 *
 * @param name name of the new buffer
 * @param numberOfObjects number of objects, which have to be allocated
 * @param objectSize number of bytes of a single object to allocate
 * @param useHostPtr true to register buffer as host-buffer, which is not activly copied by the
 *                   host to the device but instead the device pulls the data from the buffer
 *                   if needed while running the kernel
 * @param data predefined data-buffer, if new memory should be allocated
 *
 * @return false, if name already is registered, else true
 */
bool
GpuData::addBuffer(const std::string &name,
                   const uint64_t numberOfObjects,
                   const uint64_t objectSize,
                   const bool useHostPtr,
                   void* data)
{
    // precheck
    if(containsBuffer(name)) {
        return false;
    }

    // prepare worker-buffer
    WorkerBuffer newBuffer;
    newBuffer.numberOfBytes = numberOfObjects * objectSize;
    newBuffer.numberOfObjects = numberOfObjects;
    newBuffer.useHostPtr = useHostPtr;

    // allocate or set memory
    if(data == nullptr)
    {
        // fix size of the bytes to allocate, if necessary by round up to a multiple of 4096 bytes
        if(newBuffer.numberOfBytes % 4096 != 0) {
            newBuffer.numberOfBytes += 4096 - (newBuffer.numberOfBytes % 4096);
        }

        newBuffer.data = Kitsunemimi::alignedMalloc(4096, newBuffer.numberOfBytes);
    }
    else
    {
        newBuffer.data = data;
        newBuffer.allowBufferDeleteAfterClose = false;
    }

    m_buffer.try_emplace(name, newBuffer);

    return true;
}

/**
 * @brief register value
 *
 * @param name identifier name of the value
 * @param value value
 *
 * @return false, if name already is registered, else true
 */
bool
GpuData::addValue(const std::string &name,
                  const uint64_t value)
{
    // prepare worker-buffer
    WorkerBuffer newBuffer;
    newBuffer.numberOfBytes = 8;
    newBuffer.numberOfObjects = 1;
    newBuffer.useHostPtr = false;
    newBuffer.isValue = true;
    newBuffer.value = value;

    return m_buffer.try_emplace(name, newBuffer).second;
}

/**
 * @brief get worker-buffer
 *
 * @param name name of the buffer
 *
 * @return pointer to worker-buffer, if name found, else nullptr
 */
GpuData::WorkerBuffer*
GpuData::getBuffer(const std::string &name)
{
    const auto it = m_buffer.find(name);
    if(it != m_buffer.end()) {
        return &it->second;
    }

    return nullptr;
}

/**
 * @brief check if buffer-name exist
 *
 * @param name name of the buffer
 *
 * @return true, if exist, else false
 */
bool
GpuData::containsBuffer(const std::string &name)
{
    if(m_buffer.find(name) != m_buffer.end()) {
        return true;
    }

    return false;
}

/**
 * @brief get buffer
 *
 * @param name name of the buffer
 *
 * @return pointer to data, if name found, else nullptr
 */
void*
GpuData::getBufferData(const std::string &name)
{
    const auto it = m_buffer.find(name);
    if(it != m_buffer.end()) {
        return it->second.data;
    }

    return nullptr;
}

/**
 * @brief check if kernel-name exist
 *
 * @param name name of the kernel
 *
 * @return true, if exist, else false
 */
bool
GpuData::containsKernel(const std::string &name)
{
    if(m_kernel.find(name) != m_kernel.end()) {
        return true;
    }

    return false;
}

/**
 * @brief get kernel def object
 *
 * @param name name of the kernel
 *
 * @return nullptr if name not exist, else pointer to requested object
 */
GpuData::KernelDef*
GpuData::getKernel(const std::string &name)
{
    const auto it = m_kernel.find(name);
    if(it != m_kernel.end()) {
        return &it->second;
    }

    return nullptr;
}

/**
 * @brief get argument position on which the argument was binded to the kernel
 *
 * @param kernelName name of the kernel
 * @param bufferName name of the buffer
 *
 * @return position of the argument
 */
uint32_t
GpuData::getArgPosition(KernelDef* kernelDef,
                        const std::string &bufferName)
{
    const auto it = kernelDef->arguments.find(bufferName);
    if(it != kernelDef->arguments.end()) {
        return it->second;
    }

    return 0;
}

}
