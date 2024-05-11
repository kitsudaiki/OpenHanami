#ifndef CLUSTER_TEST_H
#define CLUSTER_TEST_H

#include <hanami_common/test_helper/compare_test_helper.h>

#include <string>

class LogicalHost;

class Cluster_Init_Test : public Hanami::CompareTestHelper
{
   public:
    Cluster_Init_Test();

   private:
    std::string m_clusterTemplate = "";
    LogicalHost* m_logicalHost = nullptr;

    void init();

    void initHost_test();
    void createCluster_test();
    void serialize_test();
};

#endif  // CLUSTER_TEST_H
