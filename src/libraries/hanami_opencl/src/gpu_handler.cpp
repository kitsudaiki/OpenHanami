/**
 * @file        gpu_handler.cpp
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

#include <hanami_opencl/gpu_handler.h>

#include <hanami_opencl/gpu_interface.h>
#include <hanami_common/logger.h>

namespace Hanami
{

GpuHandler::GpuHandler() {}

/**
 * @brief initialize opencl
 *
 * @param config object with config-parameter
 * @param error reference for error-output
 *
 * @return true, if creation was successful, else false
 */
bool
GpuHandler::initDevice(ErrorContainer &error)
{
    if(m_isInit) {
        return true;
    }

    LOG_DEBUG("initialize OpenCL device");

    try
    {
        // get all available opencl platforms
        cl::Platform::get(&m_platform);
        if(m_platform.empty())
        {
            error.addSolution("No OpenCL platforms found.");
            LOG_ERROR(error);
            return false;
        }

        LOG_DEBUG("number of OpenCL platforms: " + std::to_string(m_platform.size()));

        collectDevices();
        m_isInit = true;
    }
    catch(const cl::Error &err)
    {
        error.addMeesage("OpenCL error: "
                         + std::string(err.what())
                         + "("
                         + std::to_string(err.err())
                         + ")");
        LOG_ERROR(error);
        return false;
    }

    return true;
}

/**
 * @brief collect all available devices
 *
 * @param config object with config-parameter
 */
void
GpuHandler::collectDevices()
{
    // get available platforms
    for(cl::Platform &platform : m_platform)
    {
        // get available devices of the selected platform
        std::vector<cl::Device> pldev;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &pldev);
        LOG_DEBUG("number of OpenCL devices: " + std::to_string(pldev.size()));

        // select devices within the platform
        for(cl::Device &device : pldev)
        {
            // check if device is available
            if(device.getInfo<CL_DEVICE_AVAILABLE>())
            {
                /*if(false)
                {
                    // check for double precision support
                    const std::string ext = dev_it->getInfo<CL_DEVICE_EXTENSIONS>();
                    if(ext.find("cl_khr_fp64") != std::string::npos
                        && ext.find("cl_amd_fp64") != std::string::npos)
                    {
                        m_devices.push_back(*dev_it);
                        m_context = cl::Context(m_devices);
                    }
                }*/

                m_interfaces.push_back(new GpuInterface(device));
            }
        }
    }
}

}
