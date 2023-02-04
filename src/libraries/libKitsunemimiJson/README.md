# libKitsunemimiJson

## Description

This library provides the functionality to parse and handle the content of json-files. 

## Usage

**Header-file:** `libKitsunemimiJson/json_item.h`

The `JsonItem`-class is the handler for the json-file-content. The functions in the header should be really self-explaned, if something is unclear, see the following basic-example or the comments in the cpp-file.

```cpp
#include <libKitsunemimiJson/json_item.h>
#include <libKitsunemimiCommon/logger.h>

// short test-string for demonstration. 
const std::string testString(
                "{\n"
                "    \"item\": {\n"
                "        \"sub_item\": \"test_value\"\n"
                "    },\n"
                "    \"item2\": {\n"
                "        \"sub_item2\": \"something\"\n"
                "    },\n"
                "    \"loop\": [\n"
                "        {\n"
                "            \"x\": 42\n"
                "        },\n"
                "        {\n"
                "            \"x\": 42.000000\n"
                "        },\n"
                "        1234,\n"
                "        {\n"
                "            \"x\": -42.000000,\n"
                "            \"y\": true,\n"
                "            \"z\": null\n"
                "        }\n"
                "    ]\n"
                "}");



Kitsunemimi::JsonItem object;

// parse the test-string
Kitsunemimi::ErrorContainer error;
bool result = object.parse(testString, error);
// if result is true, then paring was successful
// else, error contains the error-message of the parser and can be printed with LOG_ERROR(error);


// get value
std::string value = object.get("loop").get(0).get("x").toString()
// value would contain 42 as string
// or with []-operator
value = testItem["loop"][0]["x"].toString();


// set-value
object.get("loop").get(0).get("x").setValue(1337);


// convert back into an json-file-string
std::string output = object.toString();
// output-variable would contain the same like the inital parsed testString
// but with some additional double quotes at the strin-values and maybe another 
// order of the groups and keys inside the groups and the replaced value
```

This is only a basic example. There are many more methods like insert, append, remove, etc.


IMPORTANT: The get-function has a beside the value a second argument. This is a bool-value, which says, if the get should return a copy or only a linked version. This is per default false, to `get` returns per default only a linked version for faster access. With this its possible to set values like in the example. If the original object is deleted, all with get returned linked versions become unusable. You can also do `get("value", true)` to get a fully copied version. 

IMPORTANT: The `[]`-operator is the same like get with true-flag and returns every time a copy version. So its slow and can not change values inside the tree. Be aware of this.

