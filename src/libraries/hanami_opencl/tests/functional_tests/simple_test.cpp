/**
 * @file        simple_test.cpp
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

#include "simple_test.h"

#include <hanami_opencl/gpu_handler.h>
#include <hanami_opencl/gpu_interface.h>

namespace Hanami
{

SimpleTest::SimpleTest() : Hanami::CompareTestHelper("SimpleTest") { simple_test(); }

void
SimpleTest::simple_test()
{
    const size_t testSize = 1 << 20;
    ErrorContainer error;

    // example kernel for task: c = a + b.
    const std::string kernelCode
        = "__kernel void add(\n"
          "       __global const float* a,\n"
          "       __global const float* b,\n"
          "       __global float* c,\n"
          "       ulong size"
          "       )\n"
          "{\n"
          "    size_t globalId_x = get_global_id(0);\n"
          "    int localId_x = get_local_id(0);\n"
          "    size_t globalSize_x = get_global_size(0);\n"
          "    size_t globalSize_y = get_global_size(1);\n"
          "    \n"
          "    size_t globalId = get_global_id(0) + get_global_size(0) * get_global_id(1);\n"
          "    if(get_global_id(0) == 0) { printf(\"################# size: %d\", size); }\n"
          "    if (globalId < size)\n"
          "    {\n"
          "       c[globalId] = a[globalId] + b[globalId];"
          "    }\n"
          "}\n";

    Hanami::GpuHandler oclHandler;
    assert(oclHandler.initDevice(error));

    TEST_NOT_EQUAL(oclHandler.m_interfaces.size(), 0)

    Hanami::GpuInterface* ocl = oclHandler.m_interfaces.at(0);

    // create data-object
    Hanami::GpuData data;

    data.numberOfWg.x = testSize / 128;
    data.numberOfWg.y = 1;
    data.threadsPerWg.x = 128;

    // init empty buffer
    data.addBuffer("x", testSize, sizeof(float), false);
    data.addBuffer("y", testSize, sizeof(float), false);
    data.addBuffer("z", testSize, sizeof(float), false);
    data.addValue("size", testSize);

    // convert pointer
    float* a = static_cast<float*>(data.getBufferData("x"));
    float* b = static_cast<float*>(data.getBufferData("y"));

    // write intput dat into buffer
    for (uint32_t i = 0; i < testSize; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // run
    TEST_EQUAL(ocl->initCopyToDevice(data, error), true)
    TEST_EQUAL(ocl->addKernel(data, "add", kernelCode, error), true)
    TEST_EQUAL(ocl->bindKernelToBuffer(data, "add", "x", error), true)
    TEST_EQUAL(ocl->bindKernelToBuffer(data, "add", "y", error), true)
    TEST_EQUAL(ocl->bindKernelToBuffer(data, "add", "z", error), true)
    TEST_EQUAL(ocl->bindKernelToBuffer(data, "add", "size", error), true)
    std::cout << "size outside: " << testSize << std::endl;
    // TEST_EQUAL(ocl->setLocalMemory("add", 256*256), true);
    TEST_EQUAL(ocl->run(data, "add", error), true)
    TEST_EQUAL(ocl->copyFromDevice(data, "z", error), true)

    // check result
    float* outputValues = static_cast<float*>(data.getBufferData("z"));
    TEST_EQUAL(outputValues[42], 3.0f)

    // update data on host
    for (uint32_t i = 0; i < testSize; i++) {
        a[i] = 5.0f;
    }

    // update data on device
    TEST_EQUAL(ocl->updateBufferOnDevice(data, "x", error), true)

    // second run
    TEST_EQUAL(ocl->run(data, "add", error), true)
    // copy new output back
    TEST_EQUAL(ocl->copyFromDevice(data, "z", error), true)

    // check new result
    outputValues = static_cast<float*>(data.getBufferData("z"));
    TEST_EQUAL(outputValues[42], 7.0f)

    // test memory getter
    TEST_NOT_EQUAL(ocl->getLocalMemorySize(), 0)
    TEST_NOT_EQUAL(ocl->getGlobalMemorySize(), 0)
    TEST_NOT_EQUAL(ocl->getMaxMemAllocSize(), 0)

    // test work group getter
    TEST_NOT_EQUAL(ocl->getMaxWorkGroupSize(), 0)
    TEST_NOT_EQUAL(ocl->getMaxWorkItemDimension(), 0)
    TEST_NOT_EQUAL(ocl->getMaxWorkItemSize().x, 0)
    TEST_NOT_EQUAL(ocl->getMaxWorkItemSize().y, 0)
    TEST_NOT_EQUAL(ocl->getMaxWorkItemSize().z, 0)

    // test close
    TEST_EQUAL(ocl->closeDevice(data), true)
}

}  // namespace Hanami
