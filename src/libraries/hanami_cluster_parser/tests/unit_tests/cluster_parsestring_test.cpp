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
        "    refractory_time: 1\n"
        "    neuron_cooldown: 10000000.0\n"
        "    max_connection_distance: 1\n"
        "    enable_reduction: false\n"
        "\n"
        "hexagons:\n"
        "    1,1,1\n"
        "    2,1,1\n"
        "    3,1,1\n"
        "\n"
        "inputs:\n"
        "    input_hexagon: 1,1,1\n"
        "\n"
        "outputs:\n"
        "    output_hexagon: 3,1,1\n"
        "\n");

    ClusterMeta result;
    ErrorContainer error;
    bool ret = parseCluster(&result, input, error);
    TEST_EQUAL(ret, true);
    if (ret == false) {
        LOG_ERROR(error);
    }

    TEST_EQUAL(result.version, 1);

    TEST_EQUAL(result.refractoryTime, 1);
    TEST_EQUAL(result.neuronCooldown, 10000000.0f);
    TEST_EQUAL(result.maxConnectionDistance, 1);
    TEST_EQUAL(result.enableReduction, false);

    TEST_EQUAL(result.hexagons.size(), 3);

    TEST_EQUAL(result.hexagons.at(0).x, 1);
    TEST_EQUAL(result.hexagons.at(0).y, 1);
    TEST_EQUAL(result.hexagons.at(0).z, 1);

    TEST_EQUAL(result.hexagons.at(1).x, 2);
    TEST_EQUAL(result.hexagons.at(1).y, 1);
    TEST_EQUAL(result.hexagons.at(1).z, 1);

    TEST_EQUAL(result.hexagons.at(2).x, 3);
    TEST_EQUAL(result.hexagons.at(2).y, 1);
    TEST_EQUAL(result.hexagons.at(2).z, 1);

    TEST_EQUAL(result.inputs.size(), 1);
    TEST_EQUAL(result.inputs.at(0).name, "input_hexagon");
    TEST_EQUAL(result.inputs.at(0).targetHexagonId, 0);

    TEST_EQUAL(result.outputs.size(), 1);
    TEST_EQUAL(result.outputs.at(0).name, "output_hexagon");
    TEST_EQUAL(result.outputs.at(0).targetHexagonId, 2);

    input
        = "version: 2\n"  // <-- error (invalid version-number)
          "settings:\n"
          "    refractory_time: 1\n"
          "    neuron_cooldown: 10000000.0\n"
          "    max_connection_distance: 1\n"
          "    enable_reduction: false\n"
          "\n"
          "hexagons:\n"
          "    1,1,1\n"
          "    2,1,1\n"
          "    3,1,1\n"
          "\n"
          "inputs:\n"
          "    input_hexagon: 1,1,1\n"
          "\n"
          "outputs:\n"
          "    output_hexagon: 3,1,1\n"
          "\n";

    ret = parseCluster(&result, input, error);
    TEST_EQUAL(ret, false);

    input
        = "version: 1\n"
          "settings:\n"
          "    refractory_time: 1\n"
          "    neuron_cooldown: 10000000.0\n"
          "    asdf_config: 1\n"  // <-- error (unknown keyword here)
          "    enable_reduction: false\n"
          "\n"
          "hexagons:\n"
          "    1,1,1\n"
          "    2,1,1\n"
          "    3,1,1\n"
          "\n"
          "inputs:\n"
          "    input_hexagon: 1,1,1\n"
          "\n"
          "outputs:\n"
          "    output_hexagon: 3,1,1\n"
          "\n";

    ret = parseCluster(&result, input, error);
    TEST_EQUAL(ret, false);

    input
        = "version: 1\n"
          "settings:\n"
          "    refractory_time: 1\n"
          "    neuron_cooldown: 10000000.0\n"
          "    max_connection_distance: 1\n"
          "    enable_reduction: false\n"
          "\n"
          "hexagons:\n"
          "    1,1,a\n"  // <-- error (invalid position)
          "    2,1,1\n"
          "    3,1,1\n"
          "\n"
          "inputs:\n"
          "    input_hexagon: 1,1,1\n"
          "\n"
          "outputs:\n"
          "    output_hexagon: 3,1,1\n"
          "\n";

    ret = parseCluster(&result, input, error);
    TEST_EQUAL(ret, false);

    input
        = "version: 1\n"
          "settings:\n"
          "    refractory_time: 1\n"
          "    neuron_cooldown: 10000000.0\n"
          "    max_connection_distance: 1\n"
          "    enable_reduction: false\n"
          "\n"
          "asdf:\n"  // <-- error (unknown keyword here)
          "    1,1,1\n"
          "    2,1,1\n"
          "    3,1,1\n"
          "\n"
          "inputs:\n"
          "    input_hexagon: 1,1,1\n"
          "\n"
          "outputs:\n"
          "    output_hexagon: 3,1,1\n"
          "\n";

    ret = parseCluster(&result, input, error);
    TEST_EQUAL(ret, false);

    input
        = "version: 1\n"
          "settings:\n"
          "    refractory_time: 1\n"
          "    neuron_cooldown: 10000000.0\n"
          "    max_connection_distance: 1\n"
          "    enable_reduction: false\n"
          "\n"
          "hexagons:\n"
          "    1,1,1\n"
          "    2,1,1\n"
          "    3,1,1\n"
          "\n"
          "inputs:\n"
          "    input_hexagon: 1,1,1\n"
          "\n"
          "outputs:\n"
          "    output_hexagon: 10000,1,1\n"  // <-- error (position does not exist)
          "\n";

    ret = parseCluster(&result, input, error);
    TEST_EQUAL(ret, false);

    input
        = "version: 1\n"
          "settings:\n"
          "    refractory_time: 1\n"
          "    neuron_cooldown: 10000000.0\n"
          "    max_connection_distance: 1\n"
          "    enable_reduction: false\n"
          "\n"
          "hexagons:\n"
          "    1,1,1\n"
          "    2,1,1\n"
          "    1,1,1\n"  // <-- error (position already exist)
          "\n"
          "inputs:\n"
          "    input_hexagon: 1,1,1\n"
          "\n"
          "outputs:\n"
          "    output_hexagon: 3,1,1\n"
          "\n";

    ret = parseCluster(&result, input, error);
    TEST_EQUAL(ret, false);
};

}  // namespace Hanami
