# libKitsunemimiSqlite

## Description

Simple wrapper for Sqlite3-database. Primary for easier handling the output of SELECT-queries.

## Usage by example

```cpp
#include <libKitsunemimiSqlite/sqlite.h>
#include <libKitsunemimiCommon/items/table_item.h>


Kitsunemimi::Sqlite testDB;
Kitsunemimi::ErrorContainer error;
bool ret = false;
std::string query = "";

ret = testDB.initDB("/tmp/testdb.db", error);
// if ret is false, then the error-message can be printed with:
//     std::cout<<error.toString()<<std::endl;
// this is also the case of the other cases in this example here


// CREATE-TABLE-query

query = "CREATE TABLE COMPANY("
        "ID INT PRIMARY KEY     NOT NULL,"
        "NAME           TEXT    NOT NULL,"
        "AGE            INT     NOT NULL,"
        "ADDRESS        CHAR(50),"
        "SALARY         REAL );";

ret = testDB.execSqlCommand(nullptr, query, error);

// INSERT-query

query = "INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) "
        "VALUES (1, 'Paul', 32, '{country: \"California\"}', 20000.00 ); "
        "INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY) "
        "VALUES (2, 'Allen', 25, '{country: \"Texas\"}', 15000.00 ); "
        "INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY)"
        "VALUES (3, 'Teddy', 23, '{country: \"Norway\"}', 20000.00 );"
        "INSERT INTO COMPANY (ID,NAME,AGE,ADDRESS,SALARY)"
        "VALUES (4, 'Mark', 25, '{country: \"Rich-Mond\"}', 65000.00 );";

ret = testDB.execSqlCommand(nullptr, query, error);

// SELECT-query

query = "SELECT * from COMPANY";

Kitsunemimi::TableItem resultItem;
ret = testDB.execSqlCommand(&resultItem, sql, error);
// for the SELECT-qurey a TableItem has to be given as reference. The result of the query will be written into this object

// the table-item can be printed for example like this
std::cout<<resultItem.toString()<<std::endl;
/*
The output then would look like this:

"+----+-------+-----+--------------------------+--------------+\n"
"| ID | NAME  | AGE | ADDRESS                  | SALARY       |\n"
"+====+=======+=====+==========================+==============+\n"
"| 1  | Paul  | 32  | {\"country\":\"California\"} | 20000.000000 |\n"
"+----+-------+-----+--------------------------+--------------+\n"
"| 2  | Allen | 25  | {\"country\":\"Texas\"}      | 15000.000000 |\n"
"+----+-------+-----+--------------------------+--------------+\n"
"| 3  | Teddy | 23  | {\"country\":\"Norway\"}     | 20000.000000 |\n"
"+----+-------+-----+--------------------------+--------------+\n"
"| 4  | Mark  | 25  | {\"country\":\"Rich-Mond\"}  | 65000.000000 |\n"
"+----+-------+-----+--------------------------+--------------+\n";
*/


ret = testDB.closeDB();

```