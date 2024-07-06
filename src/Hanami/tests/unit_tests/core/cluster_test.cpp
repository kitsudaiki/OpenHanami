/**
 * @file        cluster_test.cpp
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

#include "cluster_test.h"

#include <core/cluster/cluster.h>
#include <core/cluster/cluster_init.h>
#include <core/cluster/objects.h>
#include <core/io/checkpoint/buffer/buffer_io.h>
#include <core/processing/logical_host.h>
#include <core/processing/physical_host.h>
#include <hanami_hardware/host.h>

Cluster_Init_Test::Cluster_Init_Test() : Hanami::CompareTestHelper("Cluster_Init_Test")
{
    initTest();
    initHost_test();
    createCluster_test();
    serialize_test();
}

/**
 * @brief initTest
 */
void
Cluster_Init_Test::initTest()
{
    m_clusterTemplate
        = "version: 1\n"
          "settings:\n"
          "    refractory_time: 2\n"
          "    neuron_cooldown: 10000000.0\n"
          "    max_connection_distance: 3\n"
          "    enable_reduction: true\n"
          "\n"
          "bricks:\n"
          "    1,1,1\n"
          "    2,1,1\n"
          "    3,1,1\n"
          "\n"
          "inputs:\n"
          "    input_brick: \n"
          "        target: 1,1,1\n"
          "        number_of_inputs: 20\n"
          "\n"
          "outputs:\n"
          "    output_brick: \n"
          "        target: 3,1,1\n"
          "        number_of_outputs: 5\n"
          "\n";
}

/**
 * @brief initHost_test
 */
void
Cluster_Init_Test::initHost_test()
{
    bool success = false;
    Hanami::ErrorContainer error;

    // init host
    PhysicalHost physicalHost;
    physicalHost.init(error);
    m_logicalHost = physicalHost.getFirstHost();
    success = m_logicalHost != nullptr;
    TEST_EQUAL(success, true);
}

/**
 * @brief createCluster_test
 */
void
Cluster_Init_Test::createCluster_test()
{
    const std::string uuid = generateUuid().toString();
    Hanami::ErrorContainer error;
    bool success = false;

    // parse template
    Hanami::ClusterMeta parsedCluster;
    success = Hanami::parseCluster(&parsedCluster, m_clusterTemplate, error);
    TEST_EQUAL(success, true);

    // create new cluster
    Cluster newCluster(m_logicalHost);
    success = newCluster.init(parsedCluster, uuid);
    TEST_EQUAL(success, true);

    // test uuid
    TEST_EQUAL(newCluster.getUuid(), uuid);

    // test settings
    TEST_EQUAL(newCluster.clusterHeader.settings.neuronCooldown, 10000000.0);
    TEST_EQUAL(newCluster.clusterHeader.settings.refractoryTime, 2);
    TEST_EQUAL(newCluster.clusterHeader.settings.maxConnectionDistance, 3);
    TEST_EQUAL(newCluster.clusterHeader.settings.enableReduction, true);

    // test bricks
    TEST_EQUAL(newCluster.bricks.size(), 3);
    TEST_EQUAL(newCluster.bricks.at(0).header.brickId, 0);
    TEST_EQUAL(newCluster.bricks.at(1).header.brickId, 1);
    TEST_EQUAL(newCluster.bricks.at(2).header.brickId, 2);
    TEST_EQUAL(newCluster.bricks.at(0).header.isInputBrick, true);
    TEST_EQUAL(newCluster.bricks.at(0).header.isOutputBrick, false);
    TEST_EQUAL(newCluster.bricks.at(1).header.isInputBrick, false);
    TEST_EQUAL(newCluster.bricks.at(1).header.isOutputBrick, false);
    TEST_EQUAL(newCluster.bricks.at(2).header.isInputBrick, false);
    TEST_EQUAL(newCluster.bricks.at(2).header.isOutputBrick, true);

    // test neighbors of brick 0
    TEST_EQUAL(newCluster.bricks.at(0).neighbors[0], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(0).neighbors[1], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(0).neighbors[2], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(0).neighbors[3], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(0).neighbors[4], 1);
    TEST_EQUAL(newCluster.bricks.at(0).neighbors[5], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(0).neighbors[6], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(0).neighbors[7], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(0).neighbors[8], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(0).neighbors[9], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(0).neighbors[10], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(0).neighbors[11], UNINIT_STATE_32);
    success = newCluster.bricks.at(0).inputInterface != nullptr;
    TEST_EQUAL(success, true);
    success = newCluster.bricks.at(0).outputInterface == nullptr;
    TEST_EQUAL(success, true);
    success = true;
    for (uint32_t i = 0; i < NUMBER_OF_POSSIBLE_NEXT; i++) {
        success &= newCluster.bricks.at(0).possibleBrickTargetIds[i] < 3
                   && newCluster.bricks.at(0).possibleBrickTargetIds[i] != 0;
    }
    TEST_EQUAL(success, true);

    // test neighbors of brick 1
    TEST_EQUAL(newCluster.bricks.at(1).neighbors[0], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(1).neighbors[1], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(1).neighbors[2], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(1).neighbors[3], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(1).neighbors[4], 2);
    TEST_EQUAL(newCluster.bricks.at(1).neighbors[5], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(1).neighbors[6], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(1).neighbors[7], 0);
    TEST_EQUAL(newCluster.bricks.at(1).neighbors[8], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(1).neighbors[9], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(1).neighbors[10], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(1).neighbors[11], UNINIT_STATE_32);
    success = newCluster.bricks.at(1).inputInterface == nullptr;
    TEST_EQUAL(success, true);
    success = newCluster.bricks.at(1).outputInterface == nullptr;
    TEST_EQUAL(success, true);
    success = true;
    for (uint32_t i = 0; i < NUMBER_OF_POSSIBLE_NEXT; i++) {
        success &= newCluster.bricks.at(1).possibleBrickTargetIds[i] < 3
                   && newCluster.bricks.at(1).possibleBrickTargetIds[i] != 1;
    }
    TEST_EQUAL(success, true);

    // test neighbors of brick 2
    TEST_EQUAL(newCluster.bricks.at(2).neighbors[0], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(2).neighbors[1], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(2).neighbors[2], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(2).neighbors[3], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(2).neighbors[4], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(2).neighbors[5], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(2).neighbors[6], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(2).neighbors[7], 1);
    TEST_EQUAL(newCluster.bricks.at(2).neighbors[8], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(2).neighbors[9], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(2).neighbors[10], UNINIT_STATE_32);
    TEST_EQUAL(newCluster.bricks.at(2).neighbors[11], UNINIT_STATE_32);
    success = newCluster.bricks.at(2).inputInterface == nullptr;
    TEST_EQUAL(success, true);
    success = newCluster.bricks.at(2).outputInterface != nullptr;
    TEST_EQUAL(success, true);
    success = true;
    for (uint32_t i = 0; i < NUMBER_OF_POSSIBLE_NEXT; i++) {
        success &= newCluster.bricks.at(2).possibleBrickTargetIds[i] == UNINIT_STATE_32;
    }
    TEST_EQUAL(success, true);

    // test input-interfaces
    TEST_EQUAL(newCluster.inputInterfaces.size(), 1);
    TEST_EQUAL(newCluster.inputInterfaces.begin()->second.targetBrickId, 0);
    TEST_EQUAL(newCluster.inputInterfaces.begin()->second.inputNeurons.size(), 20);

    // test output-interfaces
    TEST_EQUAL(newCluster.outputInterfaces.size(), 1);
    TEST_EQUAL(newCluster.outputInterfaces.begin()->second.targetBrickId, 2);
    TEST_EQUAL(newCluster.outputInterfaces.begin()->second.outputNeurons.size(), 5);
}

/**
 * @brief serialize_test
 */
void
Cluster_Init_Test::serialize_test()
{
    const std::string uuid = generateUuid().toString();
    Hanami::ErrorContainer error;
    bool success = false;

    // parse template
    Hanami::ClusterMeta parsedCluster;
    success = Hanami::parseCluster(&parsedCluster, m_clusterTemplate, error);
    assert(success);

    // create new cluster
    Cluster baseCluster(m_logicalHost);
    assert(baseCluster.init(parsedCluster, uuid));

    // write cluster into a test-buffer
    BufferIO bufferIo;
    Hanami::DataBuffer buffer;
    TEST_EQUAL(bufferIo.writeClusterIntoBuffer(buffer, baseCluster, error), OK);

    // read cluster from the test-buffer again
    Cluster copyCluster(m_logicalHost);
    TEST_EQUAL(bufferIo.readClusterFromBuffer(copyCluster, buffer, error), OK);

    // check cluster itself
    success = copyCluster.clusterHeader == baseCluster.clusterHeader;
    TEST_EQUAL(success, true);
    TEST_EQUAL(copyCluster.bricks.size(), baseCluster.bricks.size());
    TEST_EQUAL(copyCluster.inputInterfaces.size(), baseCluster.inputInterfaces.size());
    TEST_EQUAL(copyCluster.outputInterfaces.size(), baseCluster.outputInterfaces.size());

    // check bricks
    for (uint64_t i = 0; i < copyCluster.bricks.size(); i++) {
        success = copyCluster.bricks[i].header == baseCluster.bricks[i].header;
        TEST_EQUAL(success, true);
        TEST_EQUAL(copyCluster.bricks[i].connectionBlocks.size(),
                   baseCluster.bricks[i].connectionBlocks.size());
        TEST_EQUAL(copyCluster.bricks[i].neuronBlocks.size(),
                   baseCluster.bricks[i].neuronBlocks.size());
    }
}
