#include <iostream>

#include <sql_table_test.h>
#include <hanami_common/logger.h>

int main()
{
    Hanami::initConsoleLogger(true);
    Hanami::SqlTable_Test();
    return 0;
}
