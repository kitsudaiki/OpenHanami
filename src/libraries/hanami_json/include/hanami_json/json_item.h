/**
 *  @file    json_item.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef JSON_ITEM_H
#define JSON_ITEM_H

#include <string>
#include <vector>
#include <map>

#include <hanami_common/logger.h>

namespace Kitsunemimi
{
class DataItem;

class JsonItem
{
public:
    JsonItem();
    JsonItem(const JsonItem &otherItem);
    JsonItem(DataItem* dataItem, const bool copy = false);
    JsonItem(std::map<std::string, JsonItem> &value);
    JsonItem(std::vector<JsonItem> &value);
    JsonItem(const char* value);
    JsonItem(const std::string &value);
    JsonItem(const int value);
    JsonItem(const float value);
    JsonItem(const long value);
    JsonItem(const double value);
    JsonItem(const bool value);

    ~JsonItem();

    bool parse(const std::string &input,
               ErrorContainer &error);

    // setter
    JsonItem& operator=(const JsonItem& other);
    JsonItem& operator=(const DataItem* other);
    bool setValue(const char* value);
    bool setValue(const std::string &value);
    bool setValue(const int &value);
    bool setValue(const float &value);
    bool setValue(const long &value);
    bool setValue(const double &value);
    bool setValue(const bool &value);
    bool insert(const std::string &key,
                const JsonItem &value,
                const bool force = false);
    bool append(const JsonItem &value);
    bool replaceItem(const uint32_t index,
                     const JsonItem &value);
    bool deleteContent();

    // getter
    DataItem* getItemContent() const;
    DataItem* stealItemContent();
    JsonItem operator[](const std::string key);
    JsonItem operator[](const uint32_t index);
    JsonItem get(const std::string key, const bool copy = false) const;
    JsonItem get(const uint32_t index, const bool copy = false) const;
    const std::string getString() const;
    int getInt() const;
    float getFloat() const;
    long getLong() const;
    double getDouble() const;
    bool getBool() const;
    uint64_t size() const;
    const std::vector<std::string> getKeys() const;

    // checks
    bool contains(const std::string &key) const;
    bool isValid() const;
    bool isNull() const;
    bool isMap() const;
    bool isArray() const;
    bool isValue() const;
    bool isString() const;
    bool isFloat() const;
    bool isInteger() const;
    bool isBool() const;

    // delete
    bool remove(const std::string& key);
    bool remove(const uint32_t index);

    // output
    const std::string toString(bool indent=false) const;

private:
    void clear();

    bool m_deletable = true;
    DataItem* m_content = nullptr;
};

}

#endif // JSON_ITEM_H
