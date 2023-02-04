/**
 * @file        test_blossom.h
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

#ifndef TEST_BLOSSOM_H
#define TEST_BLOSSOM_H

#include <libKitsunemimiHanamiNetwork/blossom.h>

namespace Kitsunemimi
{
namespace Hanami
{
class Session_Test;

class TestBlossom
        : public Kitsunemimi::Hanami::Blossom
{
public:
    TestBlossom(Session_Test* sessionTest);

protected:
    bool runTask(Hanami::BlossomIO &blossomIO,
                 const DataMap &context,
                 Hanami::BlossomStatus &status,
                 ErrorContainer &);

private:
    Session_Test* m_sessionTest = nullptr;
};

}  // namespace Hanami
}  // namespace Kitsunemimi

#endif // TEST_BLOSSOM_H
