#include <iostream>

#include <sql_table_test.h>
#include <hanami_common/logger.h>

int main()
{
    Kitsunemimi::initConsoleLogger(true);
    Kitsunemimi::Sakura::SqlTable_Test();
    return 0;
}
