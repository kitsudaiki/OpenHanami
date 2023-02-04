/**
 * @file        component_support.cpp
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

#include <libKitsunemimiHanamiCommon/component_support.h>

namespace Kitsunemimi
{
namespace Hanami
{

Kitsunemimi::Hanami::SupportedComponents* SupportedComponents::m_supportedComponents = nullptr;

SupportedComponents*
SupportedComponents::getInstance()
{
    if(m_supportedComponents == nullptr) {
        m_supportedComponents = new SupportedComponents();
    }

    return m_supportedComponents;
}

SupportedComponents::SupportedComponents()
{
    support[0] = false;
    support[1] = false;
    support[2] = false;
    support[3] = false;
    support[4] = false;
    support[5] = false;
}

}  // namespace Hanami
}  // namespace Kitsunemimi
