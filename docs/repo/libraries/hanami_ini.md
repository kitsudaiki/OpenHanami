# hanami_ini

!!! warning

    This documentation here is the archived version of the old readme-file of this library and is NOT up-to-date, but maybe it is still useful for some references.

## Description

This library provides the functionality to parse and handle the content of ini-files. It only
converts string, but doesn't read files from the storage.

## Usage

**Header-file:** `hanami_ini/ini_item.h`

The `IniItem`-class is the handler for the ini-file-content. The functions in the header should be
really self-explaned, if something is unclear, see the following example or the comments in the
cpp-file.

```cpp
#include <hanami_ini/ini_item.h>
#include <hanami_common/logger.h>

// short test-string for demonstration.
const std::string testString(
                "[DEFAULT]\n"
                "key = asdf.asdf\n"
                "id = 550e8400-e29b-11d4-a716-446655440000\n"
                "x = 2\n"
                "\n"
                "[xyz]\n"
                "poi_poi = 1.300000\n"
                "\n");

IniItem object;

// parse the test-string
ErrorContainer error;

bool result = object.parse(testString, error);
// if result.first is true, then paring was successful

DataItem* value = object.get("DEFAULT", "x")
// if value is a nullptr, then the group and/or item doesn't exist
// the DataItem-class comes from my library hanami_common


// get an item of the ini-file-content
int getValue = value->getInt();
// getValue now contains 2

// it could also be converted into a stirng with the toString-method
std::string getValue = object.get("DEFAULT", "x")->toString();


// set or overwrite a value
object.set("group", "item", 42, true)
// arguments: group-name, item-name, value (string, int or float), force-flag
// if the group or item doesn't exist, it will be created
// force-flag must be true, to overwrite an existing item


// remove item or group
object.removeEntry("group", "item");
object.removeGroup("group");


// convert back into an ini-file-string
std::string output = object.toString();
// output-variable would contain the same like the inital parsed testString
// but with some additional double quotes at the strin-values and maybe another
// order of the groups and keys inside the groups

```
