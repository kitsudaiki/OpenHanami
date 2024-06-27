/**
 *  @file       main.cpp
 *
 *  @author     Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright  MIT License
 */

#include <hanami_common/logger.h>
#include <obj_item_test.h>

int
main()
{
    Hanami::initConsoleLogger(true);

    ObjItem_Test test1;
}
