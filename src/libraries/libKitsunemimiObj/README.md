# libKitsunemimiObj

## Description

This library provides a simple and minimal wavefront obj-parser and creator to generate the content of such files.

Supported functions:

- vertex `v`
- normals `vn`
- textures `vt`
- points `p`
- lines `l`
- faces `f`

## Usage by example

Content of example obj-file:

```
v 2.000000 -1.000000 -1.000000
v 1.000000 -1.000000 1.000000
v -1.000000    -1.000000 1.000000
v -1.000000 -1.000000 -1.000000
v     1.000000 1.000000 -0.999999
v 0.999999 1.000000 1.000001
v -1.000000 1.000000 1.000000
v -1.000000    1.000000 -1.000000

vt -1.000000 1.000000
vt -1.000000    1.000000

vn 0.000000 -1.000000 0.000000
vn  0.000000 1.000000 0.000000
vn 1.000000 0.000000 0.000000
vn -0.000000 -0.000000 1.000000
vn -1.0000 -0.000000 -0.000000
vn 0.000000 0.000000 -1.000000

p 2
p 1

l 1 2 3 4
l 1 5 6 7

f 1//1 2//1 3//1 4//1
f 5//2 8//2   7//2 6//2
f 1//3 5//3 6//3 2//3
f 2//4 6//4 7//4 3//4
f 3//5 7//5 8//5 4//5
f 5//6   1//6 4//6 8//6
```

Example usage in code:

```cpp
#include <libKitsunemimiObj/obj_item.h>
#include <libKitsunemimiCommon/logger.h>

// parse string
Kitsunemimi::ObjItem parsedItem;
bool ret = false;
Kitsunemimi::Error error;

ret = Kitsunemimi::parseString(parsedItem, input-string, error);
// if ret is true, when it was successful
// all parsed information are stored in the parsedItem-object
// if failed the error can be printed with LOG_ERROR(error);

// create a string again
const std::string output_string = "";
ret = Kitsunemimi::convertToString(output_string, parsedItem);
// if ret is true, when it was successful
// output_string contains the now the string, which is generated from the parsedItem-object

```

