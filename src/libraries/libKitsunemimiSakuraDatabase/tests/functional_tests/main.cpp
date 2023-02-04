#include <iostream>

#include <sql_table_test.h>
#include <libKitsunemimiCommon/logger.h>

int main()
{
    Kitsunemimi::initConsoleLogger(true);
    Kitsunemimi::Sakura::SqlTable_Test();
    return 0;
}
