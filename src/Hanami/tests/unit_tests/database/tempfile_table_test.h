/**
 * @file        tempfile_table_test.h
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

#ifndef TEMPFILETABLE_TEST_H
#define TEMPFILETABLE_TEST_H

#include <hanami_common/structs.h>
#include <hanami_common/test_helper/compare_test_helper.h>

namespace Hanami
{
class SqlDatabase;
}

class TempfileTable_Test : public Hanami::CompareTestHelper
{
   public:
    TempfileTable_Test();

    void initTest();
    void createTestDb();
    void cleanupTest();

    void addTempfile_test();
    void getTempfile_test();
    void getAllTempfile_test();
    void deleteTempfile_test();
    void getRelatedResourceUuids_test();

   private:
    Hanami::UserContext m_userContext;
    Hanami::SqlDatabase* m_database = nullptr;
    const std::string m_databasePath = "/tmp/hanami_db_test";

    std::string m_testUuid1 = "";
    std::string m_testUuid2 = "";
    std::string m_testName = "";
    std::string m_testResourceUuid = "";
};

#endif  // TEMPFILETABLE_TEST_H
