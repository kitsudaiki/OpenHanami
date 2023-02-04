/**
 *  @file       data_items.h
 *
 *  @author     Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright  Apache License Version 2.0
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

#ifndef DATAITEMS_H
#define DATAITEMS_H

#include <iostream>
#include <string>
#include <string.h>
#include <vector>
#include <assert.h>
#include <map>

namespace Kitsunemimi
{
class DataArray;
class DataMap;
class DataValue;

//==================================================================================================
// DataItem
//==================================================================================================
class DataItem
{
public:
    virtual ~DataItem();

    enum dataTypes {
        UNINIT_TYPE = 0,
        VALUE_TYPE = 1,
        MAP_TYPE = 2,
        ARRAY_TYPE = 3
    };

    enum dataValueTypes {
        UNINIT_VALUE_TYPE = 0,
        STRING_TYPE = 1,
        INT_TYPE = 2,
        FLOAT_TYPE = 3,
        BOOL_TYPE = 4,
    };

    // getter
    virtual DataItem* operator[](const std::string &key) const = 0;
    virtual DataItem* operator[](const uint64_t index) const = 0;
    virtual DataItem* get(const std::string &key) const = 0;
    virtual DataItem* get(const uint64_t index) const = 0;
    virtual uint64_t size() const = 0;

    // delete
    virtual bool remove(const std::string &key) = 0;
    virtual bool remove(const uint64_t index) = 0;
    virtual void clear() = 0;

    // output
    virtual DataItem* copy() const = 0;
    virtual const std::string toString(bool indent = false,
                                       std::string* output = nullptr,
                                       uint32_t step = 0) const = 0;

    // checker
    dataTypes getType() const;
    bool isValue() const;
    bool isMap() const;
    bool isArray() const;
    bool isStringValue() const;
    bool isIntValue() const;
    bool isFloatValue() const;
    bool isBoolValue() const;

    // converter
    DataArray* toArray();
    DataMap* toMap();
    DataValue* toValue();

    // value-getter
    const std::string getString() const;
    int getInt();
    float getFloat();
    long getLong();
    double getDouble();
    bool getBool();

protected:
    dataTypes m_type = UNINIT_TYPE;
    dataValueTypes m_valueType = UNINIT_VALUE_TYPE;

    void addIndent(std::string* output,
                   const bool indent,
                   const uint32_t level) const;
};

//==================================================================================================
// DataValue
//==================================================================================================
class DataValue : public DataItem
{
public:
    DataValue();
    DataValue(const char* text);
    DataValue(const std::string &text);
    DataValue(const int value);
    DataValue(const float value);
    DataValue(const long value);
    DataValue(const double value);
    DataValue(const bool value);
    DataValue(const DataValue &other);
    ~DataValue();

    DataValue& operator=(const DataValue& other);

    // setter
    dataValueTypes getValueType();
    void setValue(const char* value);
    void setValue(const std::string &value);
    void setValue(const int &value);
    void setValue(const float &value);
    void setValue(const long &value);
    void setValue(const double &value);
    void setValue(const bool &value);

    // getter
    DataItem* operator[](const std::string &) const;
    DataItem* operator[](const uint64_t) const;
    DataItem* get(const std::string &) const;
    DataItem* get(const uint64_t) const;
    uint64_t size() const;

    // delete
    bool remove(const std::string&);
    bool remove(const uint64_t);
    void clear();

    // output
    DataItem* copy() const;
    const std::string toString(const bool indent = false,
                               std::string* output = nullptr,
                               const uint32_t = 0) const;

    // content
    union DataValueContent
    {
        char* stringValue;
        long longValue;
        double doubleValue;
        bool boolValue;
    };

    DataValueContent content;

private:
    void clearDataValue();
};

//==================================================================================================
// DataMap
//==================================================================================================
class DataMap : public DataItem
{
public:
    DataMap();
    DataMap(const DataMap &other);
    ~DataMap();

    DataMap& operator=(const DataMap& other);

    // add
    bool insert(const std::string &key,
                DataItem* value,
                bool force = false);

    // getter
    DataItem* operator[](const std::string &key) const;
    DataItem* operator[](const uint64_t index) const;
    DataItem* get(const std::string &key) const;
    DataItem* get(const uint64_t index) const;
    uint64_t size() const;

    const std::vector<std::string> getKeys() const;
    const std::vector<DataItem*> getValues() const;
    bool contains(const std::string &key) const;

    // get values by keys
    const std::string getStringByKey(const std::string &key) const;
    bool getBoolByKey(const std::string &key) const;
    int getIntByKey(const std::string &key) const;
    float getFloatByKey(const std::string &key) const;
    long getLongByKey(const std::string &key) const;
    double getDoubleByKey(const std::string &key) const;

    // delete
    bool remove(const std::string &key);
    bool remove(const uint64_t index);
    void clear();

    // output
    DataItem* copy() const;
    const std::string toString(const bool indent = false,
                               std::string* output = nullptr,
                               const uint32_t level = 0) const;

    // content
    std::map<std::string, DataItem*> map;

private:
    void clearDataMap();
};

//==================================================================================================
// DataArray
//==================================================================================================
class DataArray : public DataItem
{
public:
    DataArray();
    DataArray(const DataArray &other);
    ~DataArray();

    DataArray& operator=(const DataArray &other);

    // add
    void append(DataItem* item);

    // getter
    DataItem* operator[](const std::string &key) const;
    DataItem* operator[](const uint64_t index) const;
    DataItem* get(const std::string &) const;
    DataItem* get(const uint64_t index) const;
    uint64_t size() const;

    // delete
    bool remove(const std::string &key);
    bool remove(const uint64_t index);
    void clear();

    // output
    DataItem* copy() const;
    const std::string toString(const bool indent = false,
                               std::string* output = nullptr,
                               const uint32_t level = 0) const;

    // content
    std::vector<DataItem*> array;

private:
    void clearDataArray();
};

}  // namespace Kitsunemimi

#endif // DATAITEMS_H
