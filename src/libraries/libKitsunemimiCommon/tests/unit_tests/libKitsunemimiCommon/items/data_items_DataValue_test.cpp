/**
 *  @file    data_items_DataValue_test.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include "data_items_DataValue_test.h"
#include <libKitsunemimiCommon/items/data_items.h>

namespace Kitsunemimi
{

DataItems_DataValue_Test::DataItems_DataValue_Test()
    : Kitsunemimi::CompareTestHelper("DataItems_DataValue_Test")
{
    copy_assingment_constructor_test();
    copy_assingment_operator_test();
    operator_test();
    get_test();
    getSize_test();
    remove_test();
    clear_test();
    copy_test();
    toString_test();
    getType_test();
    isValue_isMap_isArray_test();
    toValue_toMap_toArray_test();
    getString_getInt_getFloat_getLong_getDouble_getBool_test();

    // value-exclusive
    getValueType_test();
    setValue_test();
}

/**
 * copy_assingment_constructor_test
 */
void
DataItems_DataValue_Test::copy_assingment_constructor_test()
{
    DataValue value("test");
    DataValue valueCopy = value;
    TEST_EQUAL(valueCopy.toString(), value.toString());
}

/**
 * copy_assingment_operator_test
 */
void
DataItems_DataValue_Test::copy_assingment_operator_test()
{
    DataValue value("test");
    DataValue valueCopy;
    valueCopy = value;
    TEST_EQUAL(valueCopy.toString(), value.toString());
}

/**
 * operator_test
 */
void
DataItems_DataValue_Test::operator_test()
{
    DataValue defaultValue;

    // int-access
    bool isNullptr = defaultValue[1] == nullptr;
    TEST_EQUAL(isNullptr, true);

    // string-access
    isNullptr = defaultValue["1"] == nullptr;
    TEST_EQUAL(isNullptr, true);
}

/**
 * get_test
 */
void
DataItems_DataValue_Test::get_test()
{
    DataValue defaultValue;

    // int-access
    bool isNullptr = defaultValue.get(1) == nullptr;
    TEST_EQUAL(isNullptr, true);

    // string-access
    isNullptr = defaultValue.get("1") == nullptr;
    TEST_EQUAL(isNullptr, true);
}

/**
 * getSize_test
 */
void
DataItems_DataValue_Test::getSize_test()
{
    DataValue defaultValue;
    TEST_EQUAL(defaultValue.size(), 0);
}

/**
 * remove_test
 */
void
DataItems_DataValue_Test::remove_test()
{
    DataValue defaultValue;
    TEST_EQUAL(defaultValue.remove(1), false);
}

/**
 * @brief clear_test
 */
void
DataItems_DataValue_Test::clear_test()
{
    DataValue value("test");
    value.clear();

    TEST_EQUAL(value.getType(), DataItem::VALUE_TYPE);
    TEST_EQUAL(value.getValueType(), DataItem::INT_TYPE);
    TEST_EQUAL(value.getInt(), 0);
}

/**
 * copy_test
 */
void
DataItems_DataValue_Test::copy_test()
{
    // init
    DataValue defaultValue;
    DataValue stringValue("test");
    DataValue intValue(42);
    DataValue floatValue(42.5f);
    DataValue longValue(42l);
    DataValue doubleValue(42.5);
    DataValue boolValue(true);

    // default-value
    DataValue* defaultValueCopy = dynamic_cast<DataValue*>(defaultValue.copy());
    TEST_EQUAL(std::string(defaultValue.content.stringValue),
             std::string(defaultValueCopy->content.stringValue));

    // string-value
    DataValue* stringValueCopy = dynamic_cast<DataValue*>(stringValue.copy());
    TEST_EQUAL(std::string(stringValue.content.stringValue),
             std::string(stringValueCopy->content.stringValue));

    // int-value
    DataValue* intValueCopy = dynamic_cast<DataValue*>(intValue.copy());
    TEST_EQUAL(intValue.content.longValue, intValueCopy->content.longValue);

    // float-value
    DataValue* floatValueCopy = dynamic_cast<DataValue*>(floatValue.copy());
    TEST_EQUAL(floatValue.content.doubleValue, floatValueCopy->content.doubleValue);

    // long-value
    DataValue* longValueCopy = dynamic_cast<DataValue*>(longValue.copy());
    TEST_EQUAL(longValue.content.longValue, longValueCopy->content.longValue);

    // double-value
    DataValue* doubleValueCopy = dynamic_cast<DataValue*>(doubleValue.copy());
    TEST_EQUAL(doubleValue.content.doubleValue, doubleValueCopy->content.doubleValue);

    // bool-value
    DataValue* boolValueCopy = dynamic_cast<DataValue*>(boolValue.copy());
    TEST_EQUAL(boolValue.content.boolValue, boolValueCopy->content.boolValue);

    // cleanup
    delete boolValueCopy;
    delete defaultValueCopy;
    delete stringValueCopy;
    delete intValueCopy;
    delete floatValueCopy;
    delete longValueCopy;
    delete doubleValueCopy;
}

/**
 * toString_test
 */
void
DataItems_DataValue_Test::toString_test()
{
    DataValue defaultValue;
    DataValue stringValue("test");
    DataValue intValue(42);
    DataValue floatValue(42.5f);
    DataValue longValue(42l);
    DataValue doubleValue(42.5);
    DataValue boolValue(true);

    TEST_EQUAL(defaultValue.toString(), "");
    TEST_EQUAL(stringValue.toString(), "test");
    TEST_EQUAL(intValue.toString(), "42");
    TEST_EQUAL(floatValue.toString(), "42.500000");
    TEST_EQUAL(longValue.toString(), "42");
    TEST_EQUAL(doubleValue.toString(), "42.500000");
    TEST_EQUAL(boolValue.toString(), "true");
}

/**
 * getType_test
 */
void
DataItems_DataValue_Test::getType_test()
{
    DataValue defaultValue;
    DataValue stringValue("test");
    DataValue intValue(42);
    DataValue floatValue(42.5f);
    DataValue longValue(42l);
    DataValue doubleValue(42.5);
    DataValue boolValue(true);

    TEST_EQUAL(defaultValue.getType(), DataItem::VALUE_TYPE);
    TEST_EQUAL(stringValue.getType(), DataItem::VALUE_TYPE);
    TEST_EQUAL(intValue.getType(), DataItem::VALUE_TYPE);
    TEST_EQUAL(floatValue.getType(), DataItem::VALUE_TYPE);
    TEST_EQUAL(longValue.getType(), DataItem::VALUE_TYPE);
    TEST_EQUAL(doubleValue.getType(), DataItem::VALUE_TYPE);
    TEST_EQUAL(boolValue.getType(), DataItem::VALUE_TYPE);
}

/**
 * isValue_isMap_isArray_test
 */
void
DataItems_DataValue_Test::isValue_isMap_isArray_test()
{
    DataValue defaultValue;
    TEST_EQUAL(defaultValue.isValue(), true);
    TEST_EQUAL(defaultValue.isMap(), false);
    TEST_EQUAL(defaultValue.isArray(), false);
}

/**
 * toValue_toMap_toArray_test
 */
void
DataItems_DataValue_Test::toValue_toMap_toArray_test()
{
    DataValue defaultValue;

    bool isNullptr = defaultValue.toMap() == nullptr;
    TEST_EQUAL(isNullptr, true);

    isNullptr = defaultValue.toArray() == nullptr;
    TEST_EQUAL(isNullptr, true);

    isNullptr = defaultValue.toValue() == nullptr;
    TEST_EQUAL(isNullptr, false);
}

/**
 * getString_getInt_getFloat_getLong_getDouble_getBool_test
 */
void
DataItems_DataValue_Test::getString_getInt_getFloat_getLong_getDouble_getBool_test()
{
    DataValue defaultValue;
    DataValue stringValue("test");
    DataValue intValue(42);
    DataValue floatValue(42.5f);
    DataValue longValue(42l);
    DataValue doubleValue(42.5);
    DataValue boolValue(true);

    // default-value
    TEST_EQUAL(defaultValue.getString(), "");
    TEST_EQUAL(defaultValue.getInt(), 0);
    TEST_EQUAL(defaultValue.getFloat(), 0.0f);
    TEST_EQUAL(defaultValue.getLong(), 0l);
    TEST_EQUAL(defaultValue.getDouble(), 0.0);
    TEST_EQUAL(defaultValue.getBool(), false);

    // string-value
    TEST_EQUAL(stringValue.getString(), "test");
    TEST_EQUAL(stringValue.getInt(), 0);
    TEST_EQUAL(stringValue.getFloat(), 0.0f);
    TEST_EQUAL(stringValue.getLong(), 0l);
    TEST_EQUAL(stringValue.getDouble(), 0.0);
    TEST_EQUAL(stringValue.getBool(), false);

    // int-value
    TEST_EQUAL(intValue.getString(), "");
    TEST_EQUAL(intValue.getInt(), 42);
    TEST_EQUAL(intValue.getFloat(), 0.0f);
    TEST_EQUAL(intValue.getLong(), 42l);
    TEST_EQUAL(intValue.getDouble(), 0.0);
    TEST_EQUAL(intValue.getBool(), false);

    // float-value
    TEST_EQUAL(floatValue.getString(), "");
    TEST_EQUAL(floatValue.getInt(), 0);
    TEST_EQUAL(floatValue.getFloat(), 42.5f);
    TEST_EQUAL(floatValue.getLong(), 0l);
    TEST_EQUAL(floatValue.getDouble(), 42.5);
    TEST_EQUAL(floatValue.getBool(), false);

    // long-value
    TEST_EQUAL(longValue.getString(), "");
    TEST_EQUAL(longValue.getInt(), 42);
    TEST_EQUAL(longValue.getFloat(), 0.0f);
    TEST_EQUAL(longValue.getLong(), 42l);
    TEST_EQUAL(longValue.getDouble(), 0.0);
    TEST_EQUAL(longValue.getBool(), false);

    // double-value
    TEST_EQUAL(doubleValue.getString(), "");
    TEST_EQUAL(doubleValue.getInt(), 0);
    TEST_EQUAL(doubleValue.getFloat(), 42.5f);
    TEST_EQUAL(doubleValue.getLong(), 0l);
    TEST_EQUAL(doubleValue.getDouble(), 42.5);
    TEST_EQUAL(doubleValue.getBool(), false);

    // bool-value
    TEST_EQUAL(boolValue.getString(), "");
    TEST_EQUAL(boolValue.getInt(), 0);
    TEST_EQUAL(boolValue.getFloat(), 0.0f);
    TEST_EQUAL(boolValue.getLong(), 0l);
    TEST_EQUAL(boolValue.getDouble(), 0.0);
    TEST_EQUAL(boolValue.getBool(), true);
}

/**
 * getValueType_test
 */
void
DataItems_DataValue_Test::getValueType_test()
{
    DataValue defaultValue;
    DataValue stringValue("test");
    DataValue intValue(42);
    DataValue floatValue(42.5f);
    DataValue longValue(42l);
    DataValue doubleValue(42.5);
    DataValue boolValue(true);

    TEST_EQUAL(defaultValue.getValueType(), DataItem::STRING_TYPE);
    TEST_EQUAL(stringValue.getValueType(), DataItem::STRING_TYPE);
    TEST_EQUAL(intValue.getValueType(), DataItem::INT_TYPE);
    TEST_EQUAL(floatValue.getValueType(), DataItem::FLOAT_TYPE);
    TEST_EQUAL(longValue.getValueType(), DataItem::INT_TYPE);
    TEST_EQUAL(doubleValue.getValueType(), DataItem::FLOAT_TYPE);
    TEST_EQUAL(boolValue.getValueType(), DataItem::BOOL_TYPE);
}

/**
 * setValue_test
 */
void
DataItems_DataValue_Test::setValue_test()
{
    DataValue defaultValue;

    // string-value
    defaultValue.setValue("test");
    TEST_EQUAL(defaultValue.getValueType(), DataItem::STRING_TYPE);
    TEST_EQUAL(std::string(defaultValue.content.stringValue), "test");

    // int-value
    defaultValue.setValue(42);
    TEST_EQUAL(defaultValue.getValueType(), DataItem::INT_TYPE);
    TEST_EQUAL(defaultValue.content.longValue, 42);

    // float-value
    defaultValue.setValue(42.5f);
    TEST_EQUAL(defaultValue.getValueType(), DataItem::FLOAT_TYPE);
    TEST_EQUAL(defaultValue.content.doubleValue, 42.5f);

    // long-value
    defaultValue.setValue(42l);
    TEST_EQUAL(defaultValue.getValueType(), DataItem::INT_TYPE);
    TEST_EQUAL(defaultValue.content.longValue, 42l);

    // double-value
    defaultValue.setValue(42.5);
    TEST_EQUAL(defaultValue.getValueType(), DataItem::FLOAT_TYPE);
    TEST_EQUAL(defaultValue.content.doubleValue, 42.5);

    // bool-value
    defaultValue.setValue(true);
    TEST_EQUAL(defaultValue.getValueType(), DataItem::BOOL_TYPE);
    TEST_EQUAL(defaultValue.content.boolValue, true);
}

}
