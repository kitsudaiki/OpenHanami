#ifndef SQLTABLE_TEST_H
#define SQLTABLE_TEST_H

#include <hanami_common/test_helper/compare_test_helper.h>

namespace Kitsunemimi::Sakura
{

class SqlDatabase;
class TestTable;

class SqlTable_Test
        : public Kitsunemimi::CompareTestHelper
{
public:
    SqlTable_Test();

private:
    std::string m_filePath = "";
    TestTable* m_table = nullptr;
    SqlDatabase* m_db = nullptr;
    std::string m_name1 = "user0815";
    std::string m_name2 = "other";

    void deleteFile();
    void initTest();
    void initDatabase_test();

    void initTable_test();
    void create_test();
    void get_test();
    void getAll_test();
    void update_test();
    void delete_test();
    void getNumberOfRows_test();
};

}

#endif // SQLTABLE_TEST_H
