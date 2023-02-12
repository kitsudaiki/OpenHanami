/**
 * @file       segment_parsestring_test.cpp
 *
 * @author     Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright  Apache License Version 2.0
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

#include "segment_parsestring_test.h"

#include <libKitsunemimiHanamiSegmentParser/segment_meta.h>

namespace Kitsunemimi::Hanami
{

Segment_ParseString_Test::Segment_ParseString_Test()
    : Kitsunemimi::MemoryLeakTestHelpter("Segment_ParseString_Test")
{
    parseString_test();
}

/**
 * parseString_test
 */
void
Segment_ParseString_Test::parseString_test()
{
    const std::string input("version: 1\n"
                            "segment_type: dynamic_segment\n"
                            "settings:\n"
                            "    max_synapse_sections: 100000\n"
                            "    synapse_segmentation: 10\n"
                            "    sign_neg: 0.5\n"
                            "        \n"
                            "bricks:\n"
                            "    1,1,1\n"
                            "        input: test_input\n"
                            "        number_of_neurons: 20\n"
                            "         \n "
                            "    2,1,1\n"
                            "        number_of_neurons: 10\n"
                            "          \n"
                            "    3,1,1\n"
                            "        output: test_output\n"
                            "        number_of_neurons: 5");


    SegmentMeta result;
    ErrorContainer error;

    // the parser-interface is a singleton, which is initialized on the first usage
    // so a unchecked run has to be done, for the initializing
    parseSegment(&result, input, error);

    REINIT_TEST();
    parseSegment(&result, input, error);
    CHECK_MEMORY();
}

}
