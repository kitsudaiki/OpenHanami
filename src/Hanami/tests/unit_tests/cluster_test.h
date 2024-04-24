#ifndef CLUSTER_TEST_H
#define CLUSTER_TEST_H

#include <hanami_common/test_helper/compare_test_helper.h>

#include <string>

class Cluster_Init_Test : public Hanami::CompareTestHelper
{
   public:
    Cluster_Init_Test();

   private:
    std::string m_clusterTemplate = "";

    void init();

    void createCluster_Test();
};

#endif  // CLUSTER_TEST_H
