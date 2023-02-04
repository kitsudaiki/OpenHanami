/**
 * @file       cluster_parsestring_test.cpp
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

#include "cluster_parsestring_test.h"

#include <libKitsunemimiHanamiClusterParser/cluster_meta.h>

namespace Kitsunemimi
{
namespace Hanami
{

Cluster_ParseString_Test::Cluster_ParseString_Test()
    : Kitsunemimi::CompareTestHelper("Segment_ParseString_Test")
{
    parseString_test();
}

/**
 * parseString_test
 */
void
Cluster_ParseString_Test::parseString_test()
{
    std::string input("version: 1\n"
                      "segments:\n"
                      "    input\n"
                      "        name: input1\n"
                      "        out: -> central : test_input\n"
                      " \n"
                      "    example_segment\n"
                      "        name: central\n"
                      "        out: test_output -> output\n"
                      "        out: test_output2 -> output2\n"
                      "\n"
                      "    output\n"
                      "        name: output1");

    ClusterMeta result;
    ErrorContainer error;
    bool ret = parseCluster(&result, input, error);
    TEST_EQUAL(ret, true);
    if(ret == false) {
        LOG_ERROR(error);
    }

    TEST_EQUAL(result.version, 1);
    TEST_EQUAL(result.segments.size(), 3);

    TEST_EQUAL(result.segments.at(0).type, "input");
    TEST_EQUAL(result.segments.at(0).name, "input1");
    TEST_EQUAL(result.segments.at(0).outputs.size(), 1);
    TEST_EQUAL(result.segments.at(0).outputs.at(0).sourceBrick, "x");
    TEST_EQUAL(result.segments.at(0).outputs.at(0).targetSegment, "central");
    TEST_EQUAL(result.segments.at(0).outputs.at(0).targetBrick, "test_input");

    TEST_EQUAL(result.segments.at(1).type, "example_segment");
    TEST_EQUAL(result.segments.at(1).name, "central");
    TEST_EQUAL(result.segments.at(1).outputs.size(), 2);
    TEST_EQUAL(result.segments.at(1).outputs.at(0).sourceBrick, "test_output");
    TEST_EQUAL(result.segments.at(1).outputs.at(0).targetSegment, "output");
    TEST_EQUAL(result.segments.at(1).outputs.at(0).targetBrick, "x");
    TEST_EQUAL(result.segments.at(1).outputs.at(1).sourceBrick, "test_output2");
    TEST_EQUAL(result.segments.at(1).outputs.at(1).targetSegment, "output2");
    TEST_EQUAL(result.segments.at(1).outputs.at(1).targetBrick, "x");

    TEST_EQUAL(result.segments.at(2).type, "output");
    TEST_EQUAL(result.segments.at(2).name, "output1");
    TEST_EQUAL(result.segments.at(2).outputs.size(), 0);


    input = "version: 2\n"  // <-- error
            "segments:\n"
            "    input\n"
            "        name: input1\n"
            "        out: -> central : test_input\n"
            " \n"
            "    example_segment\n"
            "        name: central\n"
            "        out: test_output -> output\n"
            "        out: test_output2 -> output2\n"
            "\n"
            "    output\n"
            "        name: output1\n";

    ret = parseCluster(&result, input, error);
    TEST_EQUAL(ret, false);


    input = "version: 1\n"
            "asdf:\n"  // <-- error
            "    input\n"
            "        name: input1\n"
            "        out: -> central : test_input\n"
            " \n"
            "    example_segment\n"
            "        name: central\n"
            "        out: test_output -> output\n"
            "        out: test_output2 -> output2\n"
            "\n"
            "    output\n"
            "        name: output1\n";

    ret = parseCluster(&result, input, error);
    TEST_EQUAL(ret, false);


    input = "version: 1\n"
            "segments:\n"
            "    input\n"
            "        asdf: input1\n"  // <-- error
            "        out: -> central : test_input\n"
            " \n"
            "    example_segment\n"
            "        name: central\n"
            "        out: test_output -> output\n"
            "        out: test_output2 -> output2\n"
            "\n"
            "    output\n"
            "        name: output1\n";

    ret = parseCluster(&result, input, error);
    TEST_EQUAL(ret, false);


    input = "version: 1\n"
            "segments:\n"
            "    input\n"
            "        name: input1\n"
            "        out: -> central : test_input\n"
            " \n"
            "    example_segment\n"
            "        name: central\n"
            "        asdf: test_output -> output\n"  // <-- error
            "        out: test_output2 -> output2\n"
            "\n"
            "    output\n"
            "        name: output1\n";

    ret = parseCluster(&result, input, error);
    TEST_EQUAL(ret, false);

    input = "version: 1\n"
            "segments:\n"
            "    input\n"
            "        name: input1\n"
            "        out: -> central : test_input\n"
            " \n"
            "    example_segment\n"
            "        name: central\n"
            "        out: test_output --> output\n"  // <-- error
            "        out: test_output2 -> output2\n"
            "\n"
            "    output\n"
            "        name: output1\n";

    ret = parseCluster(&result, input, error);
    TEST_EQUAL(ret, false);
};

}
}
