# Base infrastructure

This chapter contains basic infrastructure components, which are the same in nearly all components of the project.

## Layer-Structure

The repositories of the project separated in multiple layers for a cleaner structure:

![Layer-structure](../img/Overview_layer.drawio)

### Common-Layer

This layer contains very generic libraries, like logger, json-parser and so on. Basically there are more then enough open-source libraries to replace most of these in this layer, but I like it to have as much control over the code-base, as possible. 

Because these are very generic stuff, most of them are only under MIT-license. Only a few of them are under Apache2, because they include some external libraries again and to avoid unnecessary stupid license-trouble, the affected libraries are exceptional under Apache2 in this layer. To identify exactly, which license is used, see [Dependencies-Overview](/other/1_dependencies/).

### Sakura-Layer

The Sakura-Layer was initially invented in my side-project [SakuraTree](https://github.com/kitsudaiki/SakuraTree), where the first two libraries were created of this layer, which was the original reason for the naming of this libraries. This layer contains my own creation, with script-language, session-protocol and so on designed by myself. 

### Hanami-Layer

All repositories of this layer were created for the Hanami-AI-project. Same like the Sakura-Layer the content of this layer is all custom functionalities. Some of the libraries in this layer extend the functionalities of the Sakura-Layer, but all here was created in regard of the Hanami-AI-project, which makes the libraries of this layer more project-specific.

### Hanami-AI-Layer

The upper layer contains all executables of the project, like `AzukiHeart` and so on, and their specific libraries like `libAzukiHeart` and so on.

## **Database**

At the moment all components, which have a database, using only a `SQLite` database. It is for now the easiest and fastest installed solution. 

![Database-Layers](../img/database_layer.drawio)

### libraries

#### libKitsunemimiSqlite

This library is only a wrapper for the sqlite-database. It's primary function is to prepare the result of a select-request by packing the data into a custom table-like structure. This makes the data easier handable and directly converted into a json-string. Beside this while getting the data from the database, it tries to parse the single cells of the database before packing into the table-structure to keep type-information of all cells. This way it is also possible to write json-strings into the database, which are automatically parsed again, when reading from the database.

#### libKitsunemimiSakuraDatabase

This here is a bit similar to SQLAlchemy, but of course only in an extremly minimal version. It provided `add`, `get`, `update` and `delete` functions, which are internally converted into SQL-queries, which are then send to `libKitsunemimiSqlite`. This makes database-access very easy. Beside this it also provides functions to define the structure of a table, with types and so on, which are also internally converts into SQL when creating a table. 

#### libKitsunemimiHanamiDatabase

At the moment this library provides only a few table-presets for `libKitsunemimiSakuraDatabase`, to makes some table-fields for all tables mandatory. 

!!! info

    In the future, which library should also provide and handle version-upgrades of tables.


## **Network**

The internal networking between the components is based on an own custom network-stack.

![Network-Layers](../img/network_layer.drawio)

### libraries

#### libKitsunemimiNetwork

This library is only a abstraction-layer for basic connections. At the moment it covers TCP (with and without TLS) and Unix-Sockets. It provides a simple universal interface to make the implementation of upper layers more easy. Data, which are received over the sockets, are cached in a ring-buffer. Each connection has its own thread.
In the current state of Hanami-AI only the Unix-Sockets are used as long as all components still runs of the same host.

#### libKitsunemimiSakuraNetwork

It contains a custom session-layer-protocol with:

- initial handshake with sharing session-ids
- internal workflow to share big messages
- checking incoming protocol
- timeouts with heartbeats
- different types of error-responses
- ...

#### libKitsunemimiHanamiNetwork

This library handles primary 2 tasks:

- hold connections to all other components, which are defined in the configuration of the current component and keep them open, to avoid handshakes for each request
- place all incoming messages of a specific type into a messages-queue, where they can be collected and further transported by another thread. This is necessary in case that the message takes a few seconds for processing and to avoid timeouts in this case.

#### libKitsunemimiSakuraLang

This was originally the core-library of my automation-tool `SakuraTree`, which is deprecated. In this project here the library is reused in order to provide the base of the API-endpoints. Beside this, it validated the fields of all incoming and outgoing actions. With the help of this library, Misaki is able to generate the REST-API-documentation. 

!!! note

    The lib also provides a custom script language with the ability of using multiple threads for specific tasks. This way it is theoretically possible to implement scripts to combine multiple endpoints to one single new endpoint, but this is not used at the moment.

#### libKitsunemimiHanamiMessages

This library contains only protobuf-definitions for serialization and deserialization of messages, to be able to send them over the network-stack. 

### Messages-Paths

![Network-Layers](../img/network_layer.drawio)

Messages can have 3 different ways through the network-stack.

1. This way are the internal REST-APIs, triggered on side of the sender by `libKitsunemimiHanamiNetwork` and processed by the endpoints defined by `libKitsunemimiSakuraLang`. Each HTTP-message, which is received by the Torii, is internally forwarded over this message-path. Messages this way must contain a valid token, with is validated by Misaki, in case that the message comes from the Torii. Beside this, all fields of the message are validated based on the endpoint. This makes the message quite powerful in validation, but slow. The messages are already quite structured, so the containers of `libKitsunemimiHanamiMessages` are not necessary here.

2. This path is for internal messages, which don't need internal API-endpoint, but either are bigger messages, or messages which still need a response in order to let the thread on the sender-side block until it receive a response. This layer only takes plain binary, so the message to send has to be serialized by the messages-library.

3. Messages this way are the are most restricted, but also fast. Messages can not be bigger then `128KiB` (will be resized until version `0.2.0`) and there are no response-messages possible, which makes it a one-way messaging. 