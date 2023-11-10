/**
 * @file        gpu_interface.cpp
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

#include <hanami_common/logger.h>
#include <hanami_opencl/gpu_interface.h>

namespace Hanami
{

/**
 * @brief constructor
 *
 * @param device opencl-device
 */
GpuInterface::GpuInterface(const cl::Device& device)
{
    LOG_DEBUG("created new gpu-interface for OpenCL device: " + device.getInfo<CL_DEVICE_NAME>());

    m_device = device;
    m_context = cl::Context(m_device);
    m_queue = cl::CommandQueue(m_context, m_device);
}

/**
 * @brief destructor to close at least the device-connection
 */
GpuInterface::~GpuInterface()
{
    GpuData emptyData;
    closeDevice(emptyData);
}

/**
 * @brief copy data from host to device
 *
 * @param data object with all data
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
GpuInterface::initCopyToDevice(GpuData& data, ErrorContainer& error)
{
    LOG_DEBUG("initial data transfer to OpenCL device");

    // precheck
    if (validateWorkerGroupSize(data, error) == false) {
        return false;
    }

    // send input to device
    for (auto& [name, workerBuffer] : data.m_buffer) {
        // skip values
        if (workerBuffer.isValue) {
            continue;
        }

        LOG_DEBUG("copy data to device: " + std::to_string(workerBuffer.numberOfBytes) + " Bytes");

        // check buffer
        if (workerBuffer.numberOfBytes == 0 || workerBuffer.numberOfObjects == 0
            || workerBuffer.data == nullptr)
        {
            error.addMeesage("failed to copy data to device, because buffer with name '" + name
                             + "' has size 0 or is not initialized.");
            LOG_ERROR(error);
            return false;
        }

        // create flag for memory handling
        cl_mem_flags flags = 0;
        if (workerBuffer.useHostPtr) {
            flags = CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR;
        }
        else {
            flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
        }

        // send data or reference to device
        workerBuffer.clBuffer
            = cl::Buffer(m_context, flags, workerBuffer.numberOfBytes, workerBuffer.data);
    }

    return true;
}

/**
 * @brief add kernel to device
 *
 * @param data object with all data
 * @param kernelName name of the kernel
 * @param kernelCode kernel source-code as string
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
GpuInterface::addKernel(GpuData& data,
                        const std::string& kernelName,
                        const std::string& kernelCode,
                        ErrorContainer& error)
{
    LOG_DEBUG("add kernel with id: " + kernelName);

    // compile opencl program for found device.
    cl::Program::Sources source;
    source.push_back(kernelCode);
    cl::Program program(m_context, source);

    try {
        std::vector<cl::Device> devices = {m_device};
        program.build(devices);
    }
    catch (const cl::Error& err) {
        error.addMeesage("OpenCL compilation error\n    "
                         + program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device));
        return false;
    }

    GpuData::KernelDef def;
    def.id = kernelName;
    def.kernelCode = kernelCode;
    def.kernel = cl::Kernel(program, kernelName.c_str());

    auto ret = data.m_kernel.try_emplace(kernelName, def);
    if (ret.second == false) {
        error.addMeesage("OpenCL-Kernel with name '" + kernelName + "' already exist");
        return false;
    }

    return true;
}

/**
 * @brief bind a buffer to a kernel
 *
 * @param data data-object with the buffer to bind
 * @param kernelName, name of the kernel, which should be used
 * @param bufferName name of buffer, which should be bind to the kernel
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
GpuInterface::bindKernelToBuffer(GpuData& data,
                                 const std::string& kernelName,
                                 const std::string& bufferName,
                                 ErrorContainer& error)
{
    LOG_DEBUG("bind buffer with name '" + bufferName + "' kernel with name: '" + kernelName + "'");

    // get kernel-data
    if (data.containsKernel(kernelName) == false) {
        error.addMeesage("no kernel with name '" + kernelName + "' found");
        return false;
    }

    // check if buffer-name exist
    if (data.containsBuffer(bufferName) == false) {
        ErrorContainer error;
        error.addMeesage("no buffer with name '" + bufferName + "' found");
        return false;
    }

    // get kernel
    GpuData::KernelDef* def = data.getKernel(kernelName);
    if (def == nullptr) {
        error.addMeesage("Kernel with name '" + kernelName + "' not found.");
        return false;
    }

    // get buffer to bind to kernel
    GpuData::WorkerBuffer* buffer = &data.m_buffer[bufferName];
    if (buffer == nullptr) {
        error.addMeesage("Buffer with name '" + bufferName + "' not found.");
        return false;
    }

    // register arguments in opencl
    const uint32_t argNumber = static_cast<uint32_t>(def->arguments.size());

    LOG_DEBUG("bind buffer with name '" + bufferName + "' to argument number "
              + std::to_string(argNumber));

    try {
        if (buffer->isValue) {
            def->kernel.setArg(argNumber, static_cast<cl_ulong>(buffer->value));
        }
        else {
            def->kernel.setArg(argNumber, buffer->clBuffer);
        }
    }
    catch (const cl::Error& err) {
        error.addMeesage("OpenCL error while binding buffer to kernel: " + std::string(err.what())
                         + "(" + std::to_string(err.err()) + ")");
        LOG_ERROR(error);
        return false;
    }

    // register on which argument-position the buffer was binded
    auto ret = def->arguments.try_emplace(bufferName, argNumber);
    if (ret.second == false) {
        error.addMeesage("OpenCL-Argument with name '" + bufferName + "' is already set.");
        return false;
    }

    return true;
}

/**
 * @brief setLocalMemory
 *
 * @param data object with all data
 * @param kernelName, name of the kernel, which should be executed
 * @param localMemorySize size of the local mamory
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
GpuInterface::setLocalMemory(GpuData& data,
                             const std::string& kernelName,
                             const uint32_t localMemorySize,
                             ErrorContainer& error)
{
    // get kernel-data
    if (data.containsKernel(kernelName) == false) {
        error.addMeesage("no kernel with name '" + kernelName + "' found");
        return false;
    }

    // set arguments
    GpuData::KernelDef* def = data.getKernel(kernelName);
    def->kernel.setArg(static_cast<uint32_t>(def->arguments.size()), localMemorySize, nullptr);

    return true;
}

/**
 * @brief update data inside the buffer on the device
 *
 * @param data object with all data
 * @param bufferName name of the buffer in the kernel
 * @param error reference for error-output
 * @param numberOfObjects number of objects to copy
 * @param offset offset in buffer on device
 *
 * @return false, if copy failed of buffer is output-buffer, else true
 */
bool
GpuInterface::updateBufferOnDevice(GpuData& data,
                                   const std::string& bufferName,
                                   ErrorContainer& error,
                                   uint64_t numberOfObjects,
                                   const uint64_t offset)
{
    // check id
    if (data.containsBuffer(bufferName) == false) {
        error.addMeesage("no buffer with name '" + bufferName + "' found");
        return false;
    }

    GpuData::WorkerBuffer* buffer = data.getBuffer(bufferName);
    const uint64_t objectSize = buffer->numberOfBytes / buffer->numberOfObjects;

    // set size with value of the buffer, if size not explitely set
    if (numberOfObjects == 0) {
        numberOfObjects = buffer->numberOfObjects;
    }

    // check size
    if (offset + numberOfObjects > buffer->numberOfObjects) {
        error.addMeesage("write-position invalid");
        return false;
    }

    // update buffer
    if (buffer->useHostPtr == false && numberOfObjects != 0) {
        // write data into the buffer on the device
        if (m_queue.enqueueWriteBuffer(buffer->clBuffer,
                                       CL_FALSE,
                                       offset * objectSize,
                                       numberOfObjects * objectSize,
                                       buffer->data)
            != CL_SUCCESS)
        {
            error.addMeesage("Update buffer with name '" + bufferName + "' on gpu failed");
            return false;
        }
    }

    return true;
}

/**
 * @brief run kernel with input
 *
 * @param data input-data for the run
 * @param kernelName, name of the kernel, which should be executed
 * @param error reference for error-output
 * @param numberOfGroups if set, it override the default number of groups of the data-obj
 * @param numberOfThreadsPerGroup if set, it override the default number of threads per group
 *                                of the data-obj
 *
 * @return true, if successful, else false
 */
bool
GpuInterface::run(GpuData& data,
                  const std::string& kernelName,
                  ErrorContainer& error,
                  const uint32_t numberOfGroups,
                  const uint32_t numberOfThreadsPerGroup)
{
    cl::Event event;
    std::vector<cl::Event> events;
    events.push_back(event);

    cl::NDRange globalRange;
    cl::NDRange localRange;

    // convert ranges
    if (numberOfGroups == 0 || numberOfThreadsPerGroup == 0) {
        globalRange = cl::NDRange(data.numberOfWg.x * data.threadsPerWg.x,
                                  data.numberOfWg.y * data.threadsPerWg.y,
                                  data.numberOfWg.z * data.threadsPerWg.z);
        localRange = cl::NDRange(data.threadsPerWg.x, data.threadsPerWg.y, data.threadsPerWg.z);
    }
    else {
        globalRange = cl::NDRange(numberOfGroups * numberOfThreadsPerGroup, 1, 1);
        localRange = cl::NDRange(numberOfThreadsPerGroup, 1, 1);
    }

    // get kernel-data
    GpuData::KernelDef* def = data.getKernel(kernelName);
    if (def == nullptr) {
        error.addMeesage("no kernel with name '" + kernelName + "' found");
        return false;
    }

    try {
        // launch kernel on the device
        const uint32_t ret = m_queue.enqueueNDRangeKernel(
            def->kernel, cl::NullRange, globalRange, localRange, NULL, &events[0]);
        if (ret != CL_SUCCESS) {
            error.addMeesage("GPU-kernel failed with return-value: " + std::to_string(ret));
            return false;
        }

        cl::WaitForEvents(events);
    }
    catch (const cl::Error& err) {
        error.addMeesage("OpenCL error: " + std::string(err.what()) + "("
                         + std::to_string(err.err()) + ")");
        return false;
    }

    return true;
}

/**
 * @brief copy data of all as output marked buffer from device to host
 *
 * @param data object with all data
 * @param bufferName name of the buffer to copy into
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
GpuInterface::copyFromDevice(GpuData& data, const std::string& bufferName, ErrorContainer& error)
{
    // check id
    if (data.containsBuffer(bufferName) == false) {
        error.addMeesage("no buffer with name '" + bufferName + "' found");
        return false;
    }

    // copy result back to host
    GpuData::WorkerBuffer* buffer = data.getBuffer(bufferName);
    if (m_queue.enqueueReadBuffer(buffer->clBuffer, CL_TRUE, 0, buffer->numberOfBytes, buffer->data)
        != CL_SUCCESS)
    {
        return false;
    }

    return true;
}

/**
 * @brief GpuInterface::getDeviceName
 * @return
 */
const std::string
GpuInterface::getDeviceName()
{
    return m_device.getInfo<CL_DEVICE_NAME>();
}

/**
 * @brief close device, free buffer on device and delete all data from the data-object, which are
 *        not a null-pointer
 *
 * @param data object with all data related to the device, which will be cleared

 * @return true, if successful, else false
 */
bool
GpuInterface::closeDevice(GpuData& data)
{
    LOG_DEBUG("close OpenCL device");

    // end queue
    const cl_int ret = m_queue.finish();
    if (ret != CL_SUCCESS) {
        return false;
    }

    // free allocated memory on the host
    for (auto& [name, workerBuffer] : data.m_buffer) {
        if (workerBuffer.data != nullptr && workerBuffer.allowBufferDeleteAfterClose) {
            Hanami::alignedFree(workerBuffer.data, workerBuffer.numberOfBytes);
        }
    }

    // clear data and free memory on the device
    data.m_buffer.clear();

    return true;
}

/**
 * @brief get size of the local memory on device
 *
 * @return size of local memory on device, or 0 if no device is initialized
 */
uint64_t
GpuInterface::getLocalMemorySize()
{
    // get information
    cl_ulong size = 0;
    m_device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &size);

    return size;
}

/**
 * @brief get size of the global memory on device
 *
 * @return size of global memory on device, or 0 if no device is initialized
 */
uint64_t
GpuInterface::getGlobalMemorySize()
{
    // get information
    cl_ulong size = 0;
    m_device.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &size);

    return size;
}

/**
 * @brief get maximum memory size, which can be allocated at one time on device
 *
 * @return maximum at one time allocatable size, or 0 if no device is initialized
 */
uint64_t
GpuInterface::getMaxMemAllocSize()
{
    // get information
    cl_ulong size = 0;
    m_device.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &size);

    return size;
}

/**
 * @brief get maximum total number of work-items within a work-group
 *
 * @return maximum work-group size
 */
uint64_t
GpuInterface::getMaxWorkGroupSize()
{
    // get information
    size_t size = 0;
    m_device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &size);

    return size;
}

/**
 * @brief get maximum size of all dimensions of work-items within a work-group
 *
 * @return worker-dimension object
 */
const WorkerDim
GpuInterface::getMaxWorkItemSize()
{
    // get information
    size_t size[3];
    m_device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &size);

    // create result object
    const uint64_t dimension = getMaxWorkItemDimension();
    WorkerDim result;
    if (dimension > 0) {
        result.x = size[0];
    }
    if (dimension > 1) {
        result.y = size[1];
    }
    if (dimension > 2) {
        result.z = size[2];
    }

    return result;
}

/**
 * @brief get maximaum dimension of items
 *
 * @return number of dimensions
 */
uint64_t
GpuInterface::getMaxWorkItemDimension()
{
    // get information
    cl_uint size = 0;
    m_device.getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &size);

    return size;
}

/**
 * @brief precheck to validate given worker-group size by comparing them with the maximum values
 *        defined by the device
 *
 * @param data data-object, which also contains the worker-dimensions
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
GpuInterface::validateWorkerGroupSize(const GpuData& data, ErrorContainer& error)
{
    const uint64_t givenSize = data.threadsPerWg.x * data.threadsPerWg.y * data.threadsPerWg.z;
    const uint64_t maxSize = getMaxWorkGroupSize();

    // checko maximum size
    if (givenSize > maxSize) {
        error.addMeesage("Size of the work-group is too big. The maximum allowed is "
                         + std::to_string(maxSize) + "\n, but set was a total size of "
                         + std::to_string(givenSize));
        return false;
    }

    const WorkerDim maxDim = getMaxWorkItemSize();

    // check single dimensions
    if (data.threadsPerWg.x > maxDim.x) {
        error.addMeesage(
            "The x-dimension of the work-item size is only "
            "allowed to have a maximum of "
            + std::to_string(maxDim.x));
        return false;
    }
    if (data.threadsPerWg.y > maxDim.y) {
        error.addMeesage(
            "The y-dimension of the work-item size is only "
            "allowed to have a maximum of "
            + std::to_string(maxDim.y));
        return false;
    }
    if (data.threadsPerWg.z > maxDim.z) {
        error.addMeesage(
            "The z-dimension of the work-item size is only "
            "allowed to have a maximum of "
            + std::to_string(maxDim.z));
        return false;
    }

    return true;
}

}  // namespace Hanami
