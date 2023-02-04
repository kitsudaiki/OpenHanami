/**
 *  @file       obj_parser_test.cpp
 *
 *  @author     Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright  MIT License
 */

#include "obj_item_test.h"
#include <libKitsunemimiObj/obj_item.h>

ObjItem_Test::ObjItem_Test()
    : Kitsunemimi::CompareTestHelper("ObjItem_Test")
{
    parse_test();
    converter_test();
}

/**
 * @brief parse_test
 */
void
ObjItem_Test::parse_test()
{
    Kitsunemimi::ObjItem result;
    Kitsunemimi::ErrorContainer error;

    bool ret = Kitsunemimi::parseString(result, getTestString(), error);
    TEST_EQUAL(ret, true);

    TEST_EQUAL(result.vertizes.at(0).x, 2.0f);
    TEST_EQUAL(result.vertizes.at(6).y, 1.0f);

    TEST_EQUAL(result.normals.at(1).y, 1.0f);
    TEST_EQUAL(result.normals.at(4).x, -1.0f);

    TEST_EQUAL(result.faces.at(0).at(0).v, 1);
    TEST_EQUAL(result.faces.at(1).at(1).vn, 2);
}

/**
 * @brief converter_test
 */
void
ObjItem_Test::converter_test()
{
    Kitsunemimi::ObjItem result;
    Kitsunemimi::ErrorContainer error;
    bool ret = Kitsunemimi::parseString(result, getTestString(), error);
    TEST_EQUAL(ret, true);

    std::string convertedString = "";
    ret = Kitsunemimi::convertToString(convertedString, result);
    //std::cout<<convertedString<<std::endl;
    TEST_EQUAL(convertedString, getCompareString());
}

/**
 * @brief get a test-string
 * @return test-string
 */
const std::string
ObjItem_Test::getTestString()
{
    return std::string("v 2.000000 -1.000000 -1.000000\n"
                       "v 1.000000 -1.000000 1.000000\n"
                       "v -1.000000    -1.000000 1.000000\n"
                       "v -1.000000 -1.000000 -1.000000\n"
                       "v     1.000000 1.000000 -0.999999\n"
                       "v 0.999999 1.000000 1.000001\n"
                       "v -1.000000 1.000000 1.000000\n"
                       "v -1.000000    1.000000 -1.000000\n"
                       "\n"
                       "vt -1.000000 1.000000\n"
                       "vt -1.000000    1.000000\n"
                       "\n"
                       "vn 0.000000 -1.000000 0.000000\n"
                       "vn \t    0.000000 1.000000 0.000000\n"
                       "vn 1.000000 0.000000 0.000000\n"
                       "vn -0.000000 -0.000000 1.000000\n"
                       "vn -1.0000 -0.000000 -0.000000\n"
                       "vn 0.000000 0.000000 -1.000000\n"
                       "\n"
                       "p 2\n"
                       "p 1\n"
                       "l 1 2 3 4\n"
                       "l 1 5 6 7\n"
                       "f  \t  1//1 2//1 3//1 4//1\n"
                       "f 5//2 8//2   7//2 6//2\n"
                       "f 1//3 5//3 6//3 2//3\n"
                       "f 2//4 6//4 7//4 3//4\n"
                       "f 3//5 7//5 8//5 4//5\n"
                       "f 5//6   1//6 4//6 8//6\n");
}

/**
 * @brief ObjItem_Test::getCompareString
 * @return
 */
const std::string
ObjItem_Test::getCompareString()
{
    return std::string("v 2.000000 -1.000000 -1.000000\n"
                       "v 1.000000 -1.000000 1.000000\n"
                       "v -1.000000 -1.000000 1.000000\n"
                       "v -1.000000 -1.000000 -1.000000\n"
                       "v 1.000000 1.000000 -0.999999\n"
                       "v 0.999999 1.000000 1.000001\n"
                       "v -1.000000 1.000000 1.000000\n"
                       "v -1.000000 1.000000 -1.000000\n"
                       "vt -1.000000 1.000000\n"
                       "vt -1.000000 1.000000\n"
                       "vn 0.000000 -1.000000 0.000000\n"
                       "vn 0.000000 1.000000 0.000000\n"
                       "vn 1.000000 0.000000 0.000000\n"
                       "vn -0.000000 -0.000000 1.000000\n"
                       "vn -1.000000 -0.000000 -0.000000\n"
                       "vn 0.000000 0.000000 -1.000000\n"
                       "p 2\n"
                       "p 1\n"
                       "l 1 2 3 4\n"
                       "l 1 5 6 7\n"
                       "f 1//1 2//1 3//1 4//1\n"
                       "f 5//2 8//2 7//2 6//2\n"
                       "f 1//3 5//3 6//3 2//3\n"
                       "f 2//4 6//4 7//4 3//4\n"
                       "f 3//5 7//5 8//5 4//5\n"
                       "f 5//6 1//6 4//6 8//6\n");
}
