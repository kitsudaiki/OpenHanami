/**
 * @file        gpu_handler.h
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

#ifndef GPU_HANDLER_H
#define GPU_HANDLER_H

#include <hanami_common/logger.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl2.hpp>

namespace Hanami
{
class GpuInterface;

class GpuHandler
{
   public:
    GpuHandler();
    bool initDevice(ErrorContainer &error);

    std::vector<GpuInterface *> m_interfaces;

   private:
    bool m_isInit = false;
    std::vector<cl::Platform> m_platform;

    void collectDevices();
};

}  // namespace Hanami

#endif  // GPU_HANDLER_H
