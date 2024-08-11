# hanami_common

!!! warning

    This documentation here is the archived version of the old readme-file of this library and is NOT up-to-date, but maybe it is still useful for some references.

## Description

This library contains some functions, I need for my other projects. There are functions for
memory-handling, thread-handling, data representation and testing.

### Content Overview

#### Data-Items

_include-file:_ `hanami_common/common_items/data_items.h`

These are classes for data-representation and comparable to the dict-objects of python. The
data-items were originally the core data handling structure inside libKitsunemimiJson for
representing json-trees. Thats why the string output of these items still have json-format. The
table-items are for table styled output of information. Internally it use the data-items.

#### Tables

_include-file:_ `hanami_common/common_items/table_item.h`

This is for printing tables. Internally it use the data-items for handling the content. For example
you could parse the json-formated content of an http message via libKitsunemimiJson, use the the
resulting data-item-tree together with a header definition and print is as table. The results looks
for example like this:

```
+-----------------+---------+
| Name of column1 | column2 |
+=================+=========+
| this is a test  | k       |
+-----------------+---------+
| another test    | asdf    |
+-----------------+---------+
| x               |         |
| y               |         |
| z               |         |
+-----------------+---------+
```

#### Data buffer

_include-file:_ `hanami_common/buffer/data_buffer.h`

This is a simple buffer for binary-data. The primary advantage is the easier resizing when adding
new data. Internally it uses alligned memory, because this is necessary for the direct read- and
write-operations of the libKitsunemimiPersistence.

#### Item buffer

_include-file:_ `hanami_common/buffer/item_buffer.h`

Buffer to store objects based on the data-buffer. It handles deleted objects inside of the buffer in
a linked list to fast reallocate deleted objects at any position of the buffer.

#### Ring buffer

_include-file:_ `hanami_common/buffer/ring_buffer.h`

Ring buffer to fast continuously read and write data. Its actually used as buffer for incoming
messages in the network library libKitsunemimiNetwork for fast message-caching.

#### Stack buffer

_include-file:_ `hanami_common/buffer/stack_buffer.h`

Stack of multiple data-buffer together with a reserve-class to avoid unnecessary memory allocation.

#### Threads

_include-file:_ `hanami_common/threading/thread.h`

This class is only a collection of some thread-function like blocking and so on which I often use.
This makes the creation of threads more easy for me. Additionally this class provides the ability to
bind a new one of this thread to a specific cpu-thread.

#### Barrier

_include-file:_ `hanami_common/threading/barrier.h`

This class can block a number of threads and release automatically, if all have reached the barrier.
To avoid dead-lock, they can also released manually.

#### Tests

_include-file:_ `hanami_common/test_helper/compare_test_helper.h`,
`hanami_common/test_helper/speed_test_helper.h` and
`hanami_common/test_helper/memory_leak_test_helper.h`

These are little test-helper classes which provides basic functionallity for unit-, benchmark-, and
memory-leak-tests.

#### Statemachine

_include-file:_ `hanami_common/statemachine.h`

It's only a simple statemachine in the moment. Basically its only to change the state and check the
current state. It doesn't trigger any events after changing the state.

#### Progress-Bar

_include-file:_ `hanami_common/progress_bar.h`

Simple progress-bar for cli-output.

#### Common methods

_include-file:_ `hanami_common/common_methods/string_methods.h`,
`hanami_common/common_methods/vector_methods.h` and `hanami_common/common_methods/file_methods.h`

These contains some commonly used mehtods for strings, vectors and objects, like for example replace
substrings within a string.

#### binary-files

_include-file:_ `hanami_common/files/binary_file.h`

These are some functions to map the data-buffer of hanami_common to the storage to persist the data
of the buffer and restore them. The functions use direct read- and write-oberations to make things
faster, but this requires more custom control.

#### text-files

_include-file:_ `hanami_common/files/text_file.h`

Methods to read text files, write text files, append new text to an existing text-file, replace a
line within an existing text-file identified by a line number and repace content within an existing
text-file identified by matching the old content.

#### log-writer

_include-file:_ `hanami_common/logger.h`

Its a simple and really easy to use logger to wirte messages with timestamps to a log-file.

## Common Information aboud my projects

Here some common information about my projects and my code-styling. It's not complete and a bit
short. I will write a styling-guide for my projects, where I will write this a bit longer with more
explanation, why I make it like this.

## Usage

Its a shord overview of the most important functions of the library. If even with this something is
unclear, than please write me a mail or an issue.

### Common items

Here is a short documentation of the functions of the items. It doesn't descibe all functions,
because the most should be self explaining in the header-files, like remove, add and so on.

#### data-items

There exist three different items which all inherit the `DataItem` for generic access:

-   `DataValue`
    -   simple value-item
    -   can be string, int-value or float-value
-   `DataMap`
    -   This is a map-object for key-value-pair with a string as identifier and a DataItem-pointer
        as value
    -   data can be added with the `insert`-method
-   `DataArray`
    -   It's a internally a vector of Dataitem-pointer
    -   data can be added with the `append`-method

IMPORTANT: all getter here only return a pointer to the internal object. If you want a copy, you
have to use the `copy`-method of the objects to recursivly the data-item-tree behind the pointer.

To see all offered possebilities, which are provided by the data-items, please see the header-file
`common_items/data_items.h`. There is nearly all self-explaining, because there are basically only
getter and setter. So the following is only a minimal example:

```cpp
#include <hanami_common/common_items/data_items.h>

// init some value
DataValue stringValue("test");
DataValue intValue(42);
DataValue floatValue(42.5f);

// init and fill array
DataArray array;
array.append(stringValue.copy());
array.append(intValue.copy());
array.append(floatValue.copy());
array.append(nullptr);

// init and fill map
DataMap map;
map.insert("asdf", stringValue.copy());
map.insert("hmm", intValue.copy());
map.insert("xyz", floatValue.copy());

// add the array also to the map
map.insert("array", array.copy());

int size = map.size();
// `size` is 4 (3 values and 1 array)


std::string output = map.toString(true);
/**
`output` would look like this:

{
    "asdf": "test",
    "hmm": 42,
    "xyz": 42.500000,
    "array": [
        "test",
        42,
        42.500000,
        null
    ]
}
**/

std::string value = map.get("array")->get(2)->toValue()->toString();
// `value` would be `42.500000` as string
```

#### table-items

This is for printing informations as table on the commandline in form of this example:

```
+-----------------+---------+
| Name of column1 | column2 |
+=================+=========+
| this is a test  | k       |
+-----------------+---------+
| asdf            | hmmm    |
+-----------------+---------+
| x               |         |
| y               |         |
| z               |         |
+-----------------+---------+
```

Its primary to print informations coming from a REST-API response in form a json. So it can be
filled manually or with content which was parsed with libKitsunemimiJson (coming soon open-source).
If the content of a cell of the table contains a string with line breaks, its shown as multiple
lines line in the last row of the example.

-   manual filling of the table:

```cpp
#include <hanami_common/common_items/table_item.h>


TableItem testItem;

// add columes
// first value is the internal used name to identify the values in the content.
// the second value is the name for the output. If this is not set, it uses the internal name here too.
testItem.addColumn("column1", "Name of column1");
testItem.addColumn("column2");

testItem.addRow(std::vector<std::string>{"this is a test", "k"});
testItem.addRow(std::vector<std::string>{"asdf", "qwert"});

// convert to string
std::string output = testItem.toString();
/**
here ouput has nwo the content:

"+-----------------+---------+\n"
"| Name of column1 | column2 |\n"
"+=================+=========+\n"
"| this is a test  | k       |\n"
"+-----------------+---------+\n"
"| asdf            | hmmm    |\n"
"+-----------------+---------+\n";
**/

```

-   fill with predefined informations:

```cpp
#include <hanami_common/common_items/table_item.h>

DataArray header;
/**
content of the header in form of:

[
    {
        "inner": "column1",
        "outer": "Name of column1"
    },
    {
        "inner": "column2",
        "outer": "column2"
    }
]
**/

DataArray body;
/**
content of the body in form of:

[
    [
        "this is a test",
        "k"
    ],
    [
        "asdf",
        "qwert"
    ]
]
**/

// initialize the table-item with the predefined body and header
// The header is not required here. The columns can be added afterwards.
TableItem testItem(&body, &header);

std::string output = testItem.toString();
/**
here ouput has now the content:

"+-----------------+---------+\n"
"| Name of column1 | column2 |\n"
"+=================+=========+\n"
"| this is a test  | k       |\n"
"+-----------------+---------+\n"
"| asdf            | hmmm    |\n"
"+-----------------+---------+\n";
**/

```

The width of a column is per default limited to 500 characters. Its possible to modify this, by
calling the toString-methods with a value. For example `testItem.toString(10)` to limit the width of
a column to 10 characters. If the content of a cell of the table is longer than this value, line
breaks will be added and write it in multiple lines.

It is also possible to convert it into a vertical table. The it is returned a table with two
columns. The left is the header in vertical form and the right column is the first row of the table
in vertical form. It makes tables with only one row better readable. Example:

```cpp
TableItem testItem();
/** following content
"+-----------------+---------+\n"
"| Name of column1 | column2 |\n"
"+=================+=========+\n"
"| this is a test  | k       |\n"
"+-----------------+---------+\n"
"| asdf            | hmmm    |\n"
"+-----------------+---------+\n";
**/


std::string output = testItem.toString(100, true);
/**
first argument it the maximum size of the column and the secont is the alternative vertical mode
here ouput has now the content:

"+-----------------+----------------+\n"
"| Name of column1 | this is a test |\n"
"+-----------------+----------------+\n"
"| column2         | k              |\n"
"+-----------------+----------------+\n";

One the first row is used here for the output
**/

```

### Data Buffer

The data-buffer is only a struct with some external functions for easier byte-buffer-handling. The
internal byte-array is a alligned memory with a size of a multiple of the defined block-size. This
is necessary for direct read- and write-operations to the storage. Beside this, the struct contains
the current size of the buffer in number of bytes and number of allocated blocks. It is possible to
use the `data` as mormal byte-array for read and write operations or use the `addData_DataBuffer`
and `getBlock_DataBuffer` for access. The `addData` allocates automatically the required number of
block, if the buffer is not big enough.

```cpp
#include <hanami_common/buffer/data_buffer.h>

// initialize new data-buffer with 10 x 4KiB
DataBuffer testBuffer(10);

int value = 42;

// write data to buffer a the position pointed by the value `testBuffer.bufferPosition`
// you can set the bufferPosition directly to write at a custom location
// or write your data directly with memcpy to any position of `testBuffer.data`
bool success = addObject_DataBuffer(&testBuffer, &value);

// This example is a bit pointless, because it is the first value in the data-buffer
// It get the block with id 0 from the buffer as int-array and from with it gets the first element
int readValue = static_cast<int>(getBlock_DataBuffer(&testBuffer, 0))[0];

// write data to buffer a the position pointed by the value `testBuffer.bufferPosition`
bool success = addData_DataBuffer(&testBuffer, static_cast<void*>(value), sizeof(int);

// additional allocate 10 more block
success = allocateBlocks_DataBuffer(&testBuffer, 10);

// clear the buffer and reduce it to 10 block again
success = reset_DataBuffer(&testBuffer, 10);

```

### Threads

The usage can be explained with the following examples:

-   This is the basic look of a new class, which should run in a separate thread:

```cpp
// demo_thead.h

#include <hanami_common/thread.h>


class DemoThread
    : public Hanami::Thread
{

public:
    DemoThread() : Hanami::Thread("DemoThread");
    // each thread is given a name

    void run()
    {
        while(!m_abort)
        {
            if(m_block) {
                blockThread();
            }
            // do something
        }
    }
}
```

-   It can be called like this for example:

```cpp
#include <hanami_common/demo_thead.h>

int main()
{
    // create a thread
    DemoThread testThread();

    // start thread
    testThread.startThread();

    // bind thread to cpu-core with id 1
    testThread.bindThreadToCore(1);

    // let the thrad pause the next time the next time it start the while-loop from beginning
    testThread.initBlockThread();

    // let the thread continue its work
    testThread.continueThread();

    // stop the thread and wait until he has finished his work
    testThread.stopThread();

    return 0;
}
```

### Compare Tests

For using the unit-tests your test-class have to inherit the class `Hanami::CompareTestHelper` and
give the header fo the constructur a name for the test as string. Inside the single tests you can
than call the two macros `TEST_EQUAL(<VARIABLE_TO_CHECK> , <EXPECTED_VALUE);` and
`TEST_NOT_EQUAL(<VARIABLE_TO_CHECK> , <NOT_EXPECTED_VALUE);`. First is successful when equal and
second one is successful, when unequal.

-   After a success the result would look like this:

```cpp
------------------------------
start <name of the tests>

tests succeeded: <number of successful tests>
tests failed: <number of failed tests>
------------------------------
```

-   When a test failed, the output looks like this:

```cpp
Error in Unit-Test
   File: "<file-path where thre failed test is located>"
   Method: "<method name where thre failed test is located>"
   Line: "<line-number of the failed test inside the test-class>"
   Variable: "<name of the tested variable>"
   Should-Value: "<Value which is originally expected in the variable>"
   Is-Value: "<Value of the tested variable>"
```

HINT: this output is only as intelligent as `std::cout`, so for example if you test an
`uint8_t`-value, then cast it to int at first, to see the is-value in the error-output.

Example:

-   Header-file:

```cpp
// demo_test.h

#include <hanami_common/test_helper/compare_test_helper.h>

class Demo_Test
    : public Hanami::CompareTestHelper    // <-- connect with unit-tests
{
public:
    Demo_Test();

private:
    void some_test();
};
```

-   Source-file:

```cpp
#include "demo_test.h"

Demo_Test::Demo_Test()
    : Hanami::CompareTestHelper("Demo_Test")    // <-- give the unit-test a name
{
    some_test();    // <-- call the test-method
}

/**
 * some_test
 */
void
DataBuffer_Test::some_test()
{
    int answerForAll = 42;

    // compare two values
    // first the is-value
    // second the should-value
    TEST_EQUAL(answerForAll, 42);   // <-- single test

    // inverted test
    TEST_NOT_EQUAL(answerForAll, 0);
}
```

The result would be:

```
------------------------------
start Demo_Test

tests succeeded: 2
tests failed: 0
------------------------------
```

For more examples you could also use the tests in the test-directory of this repository.

### Memory-Leak Tests

The memory-leak-tests can check, if there are some delete-calls are missing inside your code. The
test-structure is really similar to the compare-tests. For using the memory-leak-tests your
test-class have to inherit the class `Hanami::MemoryLeakTestHelpter` and give the header fo the
constructur a name for the test as string. Inside the single tests you can than call the two macros
`REINIT_TEST();` and `CHECK_MEMORY();`. The first one has to be called every time at the beginning
of a test-case, after all required data-structures for the test are initialized. The other has to be
called at the end of the test-case. If there were some memory-leaks since the REINIT_TEST-call, a
error-message is printed.

-   After a success the result would look like this:

```cpp
------------------------------
start <name of the tests>

tests succeeded: <number of successful tests>
tests failed: <number of failed tests>
------------------------------
```

-   When a test failed, the output looks like this:

```cpp
Memory-leak detected
   File: "<file-path where thre failed test is located>"
   Method: "<method name where thre failed test is located>"
   Line: "<line-number of the failed test inside the test-class>"
   Number of missing deallocations: "<number of deallocations, which are missing>"
```

Example:

-   Header-file:

```cpp
// demo_test.h

#include <hanami_common/test_helper/memory_leak_test_helper.h>

class Demo_Test
    : public Hanami::MemoryLeakTestHelpter    // <-- connect with memory-leak-tests
{
public:
    Demo_Test();

private:
    void some_test();
};
```

-   Source-file:

```cpp
#include "demo_test.h"

Demo_Test::Demo_Test()
    : Hanami::MemoryLeakTestHelpter("Demo_Test")    // <-- give the memory-leak-test a name
{
    some_test();    // <-- call the test-method
}

/**
 * some_test
 */
void
DataBuffer_Test::some_test()
{
    // reset memory-counter
    REINIT_TEST();

    DataBuffer* testBuffer = new DataBuffer(10);
    delete testBuffer;

    // check if there were some memory-leaks since the REINIT_TEST
    CHECK_MEMORY();
}
```

The result would be:

```
------------------------------
start Demo_Test

tests succeeded: 1
tests failed: 0
------------------------------
```

For more examples you could also use the tests in the test-directory of this repository.

### Statemachine

This is really a ultra simple statemachine, so the few functions can easily explained by the
following example.

```cpp

#include <hanami_common/statemachine.h>

enum states
{
    SOURCE_STATE = 1,
    TARGET_STATE = 2,
    CHILD_STATE = 3,
    NEXT_STATE = 4,
    GO = 5,
    GOGO = 6,
};


// create statemachine
Statemachine testMachine;

// init state
statemachine.createNewState(SOURCE_STATE, "sourceState");
statemachine.createNewState(NEXT_STATE, "nextState");
statemachine.createNewState(CHILD_STATE, "childState");
statemachine.createNewState(TARGET_STATE, "targetState");

// build state-machine
statemachine.addChildState(NEXT_STATE, CHILD_STATE);
statemachine.setInitialChildState(NEXT_STATE, CHILD_STATE);

statemachine.addTransition(SOURCE_STATE, GO, NEXT_STATE);
statemachine.addTransition(NEXT_STATE, GOGO, TARGET_STATE);

// try to go to next state
bool success = testMachine.goToNextState(GO);
// success would be `true` in this case

// get curren state
const uint32_t stateId = statemachine.getCurrentStateId();
// stateId would be 4
const std::string stateName = statemachine.getCurrentStateName();
// stateName would be `nextState`

```

In the future there also should be triggered events after a state-change. Also failed states should
be added ans so on.

### Progress-Bar

```cpp

#include <hanami_common/progress_bar.h>

float progress = 0.0f;

// initalize a new progress-bar
ProgressBar* progressBar = new ProgressBar();

while(true)
{
    // updateProgress return true, if the progress reached 1.0 or more
    // this function updates the progress-bar on the terminal and alway
    // requires absolut values and not a diff to the last value
    if(progressBar->updateProgress(progress)) {
        break;
    }

    usleep(100000);
    progress += 0.02;
}

std::cout << std::endl;
```

Example-output:

```
[========================================>                                       ] 50 %
```

### Common methods

This is really minimalistic at the moment, because here are only two methods now, but there will
coming more.

#### string methods

Following functions are supported:

-   split string at a specific character into a vector of strings
-   split them into a list of substring, where each substring has a maximum size
-   replace a substring within a string

Example:

```cpp

#include <hanami_common/common_methods/string_methods.h>


std::string testString = "this is a test-string";

// split by a delimiter
std::vector<std::string> result1;
splitStringByDelimiter(result1, testString, ' ');
// the resulting list now contains ["this", "is", "a", "test-string"]

// split into max sizes
std::vector<std::string> result2;
result = splitStringByLength(result2, testString, 5);
// the resulting list now contains ["this ", "is a ", "test-", "strin", "g"]

// replace substring
std::string testString = "this is a test-string";
replaceSubstring(testString, "test", "bogus");
// variable teststring has not the content: "this is a bogus-string"

// trim string
std::string testString = "  \t  \n \r  this is a test-string  \t  \n \r  ";
trim(testString);
// variable teststring has not the content: "this is a test-string"

// to upper case string
std::string testString = "1234 this is A test-string _ !?";
toUpperCase(testString);
// variable teststring has not the content: "1234 THIS IS A TEST-STRING _ !?"

// to lower case string
std::string testString = "1234 THIS is A TEST-STRING _ !?";
toLowerCase(testString);
// variable teststring has not the content: "1234 this is a test-string _ !?"
```

#### vector methods

Here is the only function for now to clear empty string from a vector of strings.

Example:

```cpp

#include <hanami_common/common_methods/vector_methods.h>


std::vector<std::string> testVector{"x","","y","z",""};

removeEmptyStrings(&testVector);

// after this testVector only contains ["x", "y", "z"]

```

### binary-files

**Header-file:** `hanami_common/files/binary_file.h`

This file contains the class for read and write of binary-files. It use the data-buffer of
hanami_common as cache for all operations. The operations using posix-method with direct-flag to
skip the page-chache of the linux-kernel. This makes operations with big block a bit faster because
the data are less often copied. This has the result, that all read and write operations are
synchronized.

This results in the requirement, that segments to read from storage or write to storage should be as
big as possible or else the latency makes the whole thing very very slow. The class should be run in
an extra thread, with handle all operations and makes the whole sync asynchon again.

There are only 4 operations at the moment:

-   allocate more memory on the storage to make the file bigger
-   read the current size of the file from the storage (for the case you open an existing file)
-   wirte data from the buffer to the file
-   read data from the file into the buffer

All operations return only a bool-value, which say if it was successful or not.

HINT: The data-buffer will be not be binded anymore in the next version.

Example:

```cpp
#include <hanami_common/files/binary_file.h>

std::string filePath = "/tmp/testfile.bin";
Common::DataBuffer buffer(2);

// write somethin into the buffer
int testvalue = 42;
buffer.addData(&testvalue);

// create binary file and bind buffer
BinaryFile binaryFile(m_filePath);

// allocate 4 x 4 KiB (4 blocks)
binaryFile.allocateStorage(4,       // <-- number blocks
                           4096);   // <-- size of a single block

// write data to the storage
binaryFile.writeSegment(&buffer,   // <-- source-buffer
                        0,         // <-- startblock of write-oberation within the file
                        1,         // <-- number of blocks (each 4 KiB) to write
                        0)         // <-- startblock of the data within the buffer

// read data to the storage
binaryFile.readSegment(&buffer,    // <-- target-buffer
                       0,          // <-- startblock of the data within the file
                       1,          // <-- number of blocks (each 4 KiB) to write
                       1)          // <-- startblock of write-oberation within the buffer

// close file
binaryFile.closeFile()
```

### text-files

**Header-file:** `hanami_common/files/text_file.h`

Every action open and close the text-file. With this I don't need to handle an additional
object-instance and operations on a text-file are really rare compared to working on a binary-file,
so the addional time consumption for open and close the file has no special meaning for the
performance.

All methods return a pair of bool-value as first element and a string-value as second element. The
bool-value says, if the call was successful or not. When successful, the string-value contains the
result, or if not successful, the string contains an error-message.

Little example:

```cpp
#include <hanami_common/files/text_file.h>
#include <hanami_common/logger.h>

std::string filePath = "/tmp/textfile.txt";

std::string content = "this is a test\n"
                      "and this is a second line";

std::pair<bool, std::string> ret;
ErrorContainer error;

// write text to file
bool writeResult = writeFile(filePath,
                             content,
                             error,
                             false);        // <-- force-flag,
                                            //     with false it fails if file already existing

// add new text to the file
bool appendResult = appendText(filePath,
                               "\nand a third line",
                               error);

// read updated file
std::pair<bool, std::string> readResult = readFile(filePath, error);
// readResult.second would now contans:
//
// "this is a test\n"
// "and this is a second line\n"
// "and a third line";

```

### log-writer

**Header-file:** `hanami_common/logger.h`

Its a simple class to write log-messages together with a timestamp one after another to a log-file.
It only has to be initialized at the beginning of the program and can be used at every point in the
same code. When want to add an entry to the log, you don't need to check, if the logger is
initialized.

IMPORTANT: Adding entries to the log is thread-save, but initializing and closing the logger is NOT.
This is normally no problem, but I only mention it, to be sure that you know this. It is not save to
init or close the logger, while other threads with log-calls are running!

Initializing at the anytime somewhere in your code.

```cpp
#include <hanami_common/logger.h>

// initializing logger to write into a file
bool ret1 = Hanami::Persistence::initFileLogger("/tmp", "testlog", true);
// arguments:
//      first argument: directory-path
//      second argument: base file name
//      third argument: true to enable debug-output. if false only output of info, warning and error
//
// result:
//      true, if initializing was successfule, else false

// initializing logger to write log-messages on the console output
Hanami::Persistence::initConsoleLogger(true);
// argument: true to enable debug-output. if false only output of info, warning and error

```

Using the logger somewhere else in your code. You only need to import the header and then call the
log-methods. Like already mentioned, there is no check necessary, if the logger is initialized or
not. See following example:

```cpp
#include <hanami_common/logger.h>

LOG_DEBUG("debug-message");
LOG_INFO("info-message");
LOG_WARNING("warning-message");

// error-messages are handle by a container, which is printed as table which logging
//     it also allows an additional field for a possible solution for easier debugging
Hanami::ErrorContainer error;
error.addMeesage("some error");
error.addSolution("do nothing");
LOG_ERROR("error-message");

/**
The log-file would look like this:

2019-9-7 22:54:1 ERROR:
+---------------------+------------+
| Error-Message Nr. 0 | some error |
+---------------------+------------+
| Possible Solution   | do nothing |
+---------------------+------------+
2019-9-7 22:54:1 WARNING: warning-message
2019-9-7 22:54:1 DEBUG: debug-message
2019-9-7 22:54:1 INFO: info-message
*/

```
