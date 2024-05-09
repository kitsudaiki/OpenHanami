#include "cluster_test.h"

#include <core/cluster/cluster.h>
#include <core/cluster/cluster_init.h>
#include <core/processing/logical_host.h>
#include <core/processing/objects.h>
#include <core/processing/physical_host.h>
#include <hanami_hardware/host.h>

Cluster_Init_Test::Cluster_Init_Test() : Hanami::CompareTestHelper("Cluster_Init_Test")
{
    init();
    createCluster_Test();
}

void
Cluster_Init_Test::init()
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

void
Cluster_Init_Test::createCluster_Test()
{
    const std::string uuid = generateUuid().toString();
    Hanami::ErrorContainer error;
    bool success = false;
    LogicalHost* host = nullptr;

    // init host
    // PhysicalHost physicalHost;
    // physicalHost.init(error);
    // host = physicalHost.getFirstHost();
    // success = host != nullptr;
    // TEST_EQUAL(success, true);

    // parse template
    Hanami::ClusterMeta parsedCluster;
    success = Hanami::parseCluster(&parsedCluster, m_clusterTemplate, error);
    TEST_EQUAL(success, true);

    // create new cluster
    Cluster* newCluster = new Cluster(host);
    success = newCluster->init(parsedCluster, uuid);
    TEST_EQUAL(success, true);

    // test uuid
    TEST_EQUAL(newCluster->getUuid(), uuid);

    // test settings
    TEST_EQUAL(newCluster->clusterHeader.settings.neuronCooldown, 10000000.0);
    TEST_EQUAL(newCluster->clusterHeader.settings.refractoryTime, 2);
    TEST_EQUAL(newCluster->clusterHeader.settings.maxConnectionDistance, 3);
    TEST_EQUAL(newCluster->clusterHeader.settings.enableReduction, true);

    // test bricks
    TEST_EQUAL(newCluster->bricks.size(), 3);
    TEST_EQUAL(newCluster->bricks.at(0).header.brickId, 0);
    TEST_EQUAL(newCluster->bricks.at(1).header.brickId, 1);
    TEST_EQUAL(newCluster->bricks.at(2).header.brickId, 2);
    TEST_EQUAL(newCluster->bricks.at(0).header.isInputBrick, true);
    TEST_EQUAL(newCluster->bricks.at(0).header.isOutputBrick, false);
    TEST_EQUAL(newCluster->bricks.at(1).header.isInputBrick, false);
    TEST_EQUAL(newCluster->bricks.at(1).header.isOutputBrick, false);
    TEST_EQUAL(newCluster->bricks.at(2).header.isInputBrick, false);
    TEST_EQUAL(newCluster->bricks.at(2).header.isOutputBrick, true);

    // test neighbors of brick 0
    TEST_EQUAL(newCluster->bricks.at(0).neighbors[0], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(0).neighbors[1], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(0).neighbors[2], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(0).neighbors[3], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(0).neighbors[4], 1);
    TEST_EQUAL(newCluster->bricks.at(0).neighbors[5], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(0).neighbors[6], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(0).neighbors[7], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(0).neighbors[8], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(0).neighbors[9], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(0).neighbors[10], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(0).neighbors[11], UNINIT_STATE_32);
    success = newCluster->bricks.at(0).inputInterface != nullptr;
    TEST_EQUAL(success, true);
    success = newCluster->bricks.at(0).outputInterface == nullptr;
    TEST_EQUAL(success, true);
    success = true;
    for (uint32_t i = 0; i < NUMBER_OF_POSSIBLE_NEXT; i++) {
        success &= newCluster->bricks.at(0).possibleBrickTargetIds[i] < 3
                   && newCluster->bricks.at(0).possibleBrickTargetIds[i] != 0;
    }
    TEST_EQUAL(success, true);

    // test neighbors of brick 1
    TEST_EQUAL(newCluster->bricks.at(1).neighbors[0], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(1).neighbors[1], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(1).neighbors[2], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(1).neighbors[3], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(1).neighbors[4], 2);
    TEST_EQUAL(newCluster->bricks.at(1).neighbors[5], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(1).neighbors[6], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(1).neighbors[7], 0);
    TEST_EQUAL(newCluster->bricks.at(1).neighbors[8], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(1).neighbors[9], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(1).neighbors[10], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(1).neighbors[11], UNINIT_STATE_32);
    success = newCluster->bricks.at(1).inputInterface == nullptr;
    TEST_EQUAL(success, true);
    success = newCluster->bricks.at(1).outputInterface == nullptr;
    TEST_EQUAL(success, true);
    success = true;
    for (uint32_t i = 0; i < NUMBER_OF_POSSIBLE_NEXT; i++) {
        success &= newCluster->bricks.at(1).possibleBrickTargetIds[i] < 3
                   && newCluster->bricks.at(1).possibleBrickTargetIds[i] != 1;
    }
    TEST_EQUAL(success, true);

    // test neighbors of brick 2
    TEST_EQUAL(newCluster->bricks.at(2).neighbors[0], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(2).neighbors[1], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(2).neighbors[2], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(2).neighbors[3], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(2).neighbors[4], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(2).neighbors[5], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(2).neighbors[6], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(2).neighbors[7], 1);
    TEST_EQUAL(newCluster->bricks.at(2).neighbors[8], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(2).neighbors[9], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(2).neighbors[10], UNINIT_STATE_32);
    TEST_EQUAL(newCluster->bricks.at(2).neighbors[11], UNINIT_STATE_32);
    success = newCluster->bricks.at(2).inputInterface == nullptr;
    TEST_EQUAL(success, true);
    success = newCluster->bricks.at(2).outputInterface != nullptr;
    TEST_EQUAL(success, true);
    success = true;
    for (uint32_t i = 0; i < NUMBER_OF_POSSIBLE_NEXT; i++) {
        success &= newCluster->bricks.at(2).possibleBrickTargetIds[i] == UNINIT_STATE_32;
    }
    TEST_EQUAL(success, true);

    // test input-interfaces
    TEST_EQUAL(newCluster->inputInterfaces.size(), 1);
    TEST_EQUAL(newCluster->inputInterfaces.begin()->second.numberOfInputNeurons, 20);
    TEST_EQUAL(newCluster->inputInterfaces.begin()->second.targetBrickId, 0);
    success = newCluster->inputInterfaces.begin()->second.inputNeurons != nullptr;
    TEST_EQUAL(success, true);

    // test output-interfaces
    TEST_EQUAL(newCluster->outputInterfaces.size(), 1);
    TEST_EQUAL(newCluster->outputInterfaces.begin()->second.numberOfOutputNeurons, 5);
    TEST_EQUAL(newCluster->outputInterfaces.begin()->second.targetBrickId, 2);
    success = newCluster->outputInterfaces.begin()->second.outputNeurons != nullptr;
    TEST_EQUAL(success, true);
}
