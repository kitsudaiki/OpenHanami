#include <hanami_common/logger.h>
#include <sql_table_test.h>

#include <iostream>

int
main()
{
    Hanami::initConsoleLogger(true);
    Hanami::SqlTable_Test();
    return 0;
}
