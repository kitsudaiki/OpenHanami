/**
 *  @file    data_items_DataMap_test.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef DATAITEMS_DATAMAP_TEST_H
#define DATAITEMS_DATAMAP_TEST_H

#include <libKitsunemimiCommon/test_helper/compare_test_helper.h>

namespace Kitsunemimi
{
class DataMap;

class DataItems_DataMap_Test
        : public Kitsunemimi::CompareTestHelper
{
public:
    DataItems_DataMap_Test();

private:
    void copy_assingment_constructor_test();
    void copy_assingment_operator_test();
    void operator_test();
    void get_test();
    void getSize_test();
    void remove_test();
    void clear_test();
    void copy_test();
    void toString_test();
    void getType_test();
    void isValue_isMap_isArray_test();
    void toValue_toMap_toArray_test();
    void getString_getInt_getFloat_test();

    void insert_test();
    void getStringByKey_getIntByKey_getFloatByKey_test();
    void getKeys_test();
    void getValues_test();
    void contains_test();

    DataMap initTestObject();
};

}  // namespace Kitsunemimi

#endif // DATAITEMS_DATAMAP_TEST_H
