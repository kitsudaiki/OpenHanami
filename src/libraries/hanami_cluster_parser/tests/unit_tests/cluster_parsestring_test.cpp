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

#include <hanami_cluster_parser/cluster_meta.h>

namespace Hanami
{

Cluster_ParseString_Test::Cluster_ParseString_Test()
    : Hanami::CompareTestHelper("Cluster_ParseString_Test")
{
    parseString_test();
}

/**
 * parseString_test
 */
void
Cluster_ParseString_Test::parseString_test()
{
    std::string input(
        "version: 1\n"
        "settings:\n"
        "   neuron_cooldown: 10000000.0\n"
        "   refractory_time: 1\n"
        "   max_connection_distance: 1\n"
        "   enable_reduction: false\n"
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

    ClusterMeta result;
    ErrorContainer error;
    bool ret = parseCluster(&result, input, error);
    TEST_EQUAL(ret, true);
    if (ret == false) {
        LOG_ERROR(error);
    }

    TEST_EQUAL(result.version, 1);

    // TEST_EQUAL(result.maxSynapseSections, 100000);
    // TEST_EQUAL(result.synapseSegmentation, 10);
    // TEST_EQUAL(result.signNeg, 0.5);

    TEST_EQUAL(result.bricks.size(), 3);

    TEST_EQUAL(result.bricks.at(0).position.x, 1);
    TEST_EQUAL(result.bricks.at(0).position.y, 1);
    TEST_EQUAL(result.bricks.at(0).position.z, 1);
    TEST_EQUAL(result.bricks.at(0).numberOfNeurons, 20);
    TEST_EQUAL(result.bricks.at(0).name, "test_input");
    TEST_EQUAL(result.bricks.at(0).type, INPUT_BRICK_TYPE);

    TEST_EQUAL(result.bricks.at(1).position.x, 2);
    TEST_EQUAL(result.bricks.at(1).position.y, 1);
    TEST_EQUAL(result.bricks.at(1).position.z, 1);
    TEST_EQUAL(result.bricks.at(1).numberOfNeurons, 10);
    TEST_EQUAL(result.bricks.at(1).name, "");
    TEST_EQUAL(result.bricks.at(1).type, CENTRAL_BRICK_TYPE);

    TEST_EQUAL(result.bricks.at(2).position.x, 3);
    TEST_EQUAL(result.bricks.at(2).position.y, 1);
    TEST_EQUAL(result.bricks.at(2).position.z, 1);
    TEST_EQUAL(result.bricks.at(2).numberOfNeurons, 5);
    TEST_EQUAL(result.bricks.at(2).name, "test_output");
    TEST_EQUAL(result.bricks.at(2).type, OUTPUT_BRICK_TYPE);

    input
        = "version: 2\n"  // <-- error
          "settings:\n"
          "    max_synapse_sections: 100000\n"
          "    synapse_clusteration: 10\n"
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
          "        number_of_neurons: 5\n";

    ret = parseCluster(&result, input, error);
    TEST_EQUAL(ret, false);

    input
        = "version: 1\n"
          "settings:\n"
          "    max_synapse_sections: 100000\n"  // <-- error
          "    asdf_clusteration: 10\n"
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
          "        number_of_neurons: 5\n";

    ret = parseCluster(&result, input, error);
    TEST_EQUAL(ret, false);

    input
        = "version: 1\n"
          "settings:\n"
          "    max_synapse_sections: 100000\n"
          "    synapse_clusteration: 10\n"
          "    sign_neg: 123\n"
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
          "        number_of_neurons: 5\n";

    ret = parseCluster(&result, input, error);
    TEST_EQUAL(ret, false);

    input
        = "version: 1\n"
          "settings:\n"
          "    max_synapse_sections: 100000\n"
          "    synapse_clusteration: 10\n"
          "    sign_neg: 0.5\n"
          "        \n"
          "bricks:\n"
          "    1,1,a\n"  // <-- error
          "        input: test_input\n"
          "        number_of_neurons: 20\n"
          "         \n "
          "    2,1,1\n"
          "        number_of_neurons: 10\n"
          "          \n"
          "    3,1,1\n"
          "        output: test_output\n"
          "        number_of_neurons: 5\n";

    ret = parseCluster(&result, input, error);
    TEST_EQUAL(ret, false);

    input
        = "version: 1\n"
          "settings:\n"
          "    max_synapse_sections: 100000\n"
          "    synapse_clusteration: 10\n"
          "    sign_neg: 0.5\n"
          "        \n"
          "asdf:\n"  // <-- error
          "    1,1,1\n"
          "        input: test_input\n"
          "        number_of_neurons: 20\n"
          "         \n "
          "    2,1,1\n"
          "        number_of_neurons: 10\n"
          "          \n"
          "    3,1,1\n"
          "        output: test_output\n"
          "        number_of_neurons: 5\n";

    ret = parseCluster(&result, input, error);
    TEST_EQUAL(ret, false);

    input
        = "version: 1\n"
          "settings:\n"
          "    max_synapse_sections: 100000\n"
          "    synapse_clusteration: 10\n"
          "    sign_neg: 0.5\n"
          "        \n"
          "bricks:\n"
          "    1,1,1\n"
          "        asdf: test_input\n"  // <-- error
          "        number_of_neurons: 20\n"
          "         \n "
          "    2,1,1\n"
          "        number_of_neurons: 10\n"
          "          \n"
          "    3,1,1\n"
          "        output: test_output\n"
          "        number_of_neurons: 5\n";

    ret = parseCluster(&result, input, error);
    TEST_EQUAL(ret, false);

    input
        = "version: 1\n"
          "settings:\n"
          "    max_synapse_sections: 100000\n"
          "    synapse_clusteration: 10\n"
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
          "        asdf: test_output\n"  // <-- error
          "        number_of_neurons: 5\n";

    ret = parseCluster(&result, input, error);
    TEST_EQUAL(ret, false);

    input
        = "version: 1\n"
          "settings:\n"
          "    max_synapse_sections: 100000\n"
          "    synapse_clusteration: 10\n"
          "    sign_neg: 0.5\n"
          "        \n"
          "bricks:\n"
          "    1,1,1\n"
          "        input: test_input\n"
          "        number_of_neurons: 20\n"
          "         \n "
          "    2,1,1\n"
          "        number_of_neurons: a\n"  // <-- error
          "          \n"
          "    3,1,1\n"
          "        output: test_output\n"
          "        number_of_neurons: 5\n";

    ret = parseCluster(&result, input, error);
    TEST_EQUAL(ret, false);
};

}  // namespace Hanami
