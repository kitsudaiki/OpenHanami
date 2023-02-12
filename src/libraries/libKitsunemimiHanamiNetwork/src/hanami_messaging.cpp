/**
 * @file        messaging_controller.cpp
 *
 * @author      Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright   Apache License Version 2.0
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

#include <libKitsunemimiHanamiNetwork/hanami_messaging.h>
#include <callbacks.h>
#include <items/item_methods.h>

#include <libKitsunemimiSakuraNetwork/session.h>
#include <libKitsunemimiSakuraNetwork/session_controller.h>

#include <libKitsunemimiHanamiCommon/config.h>
#include <libKitsunemimiHanamiCommon/component_support.h>
#include <libKitsunemimiHanamiNetwork/hanami_messaging_client.h>
#include <libKitsunemimiHanamiNetwork/blossom.h>

#include <libKitsunemimiCommon/logger.h>
#include <libKitsunemimiCommon/files/text_file.h>
#include <libKitsunemimiCommon/buffer/stack_buffer.h>
#include <libKitsunemimiConfig/config_handler.h>
#include <libKitsunemimiCrypto/common.h>
#include <libKitsunemimiJwt/jwt.h>

#include <../../libKitsunemimiHanamiMessages/protobuffers/shiori_messages.proto3.pb.h>
#include <../../libKitsunemimiHanamiMessages/message_sub_types.h>

namespace Kitsunemimi::Hanami
{

Kitsunemimi::Sakura::SessionController* HanamiMessaging::m_sessionController = nullptr;

HanamiMessaging* HanamiMessaging::m_messagingController = nullptr;

/**
 * @brief constructor
 */
HanamiMessaging::HanamiMessaging()
{
    m_sessionController = new Sakura::SessionController(&sessionCreateCallback,
                                                        &sessionCloseCallback,
                                                        &errorCallback);
}

/**
 * @brief destructor
 */
HanamiMessaging::~HanamiMessaging() {}

/**
 * @brief callback, which is triggered by error-logs
 *
 * @param errorMessage error-message to send to shiori
 */
void
handleErrorCallback(const std::string &errorMessage)
{
    HanamiMessaging::getInstance()->sendGenericErrorMessage(errorMessage);
}

/**
 * @brief fill overview with all configured components
 */
void
HanamiMessaging::fillSupportOverview()
{
    bool success = false;

    SupportedComponents* supportedComponents = SupportedComponents::getInstance();

    if(GET_STRING_CONFIG("kyouko", "address", success) != "") {
        supportedComponents->support[KYOUKO] = true;
    }
    if(GET_STRING_CONFIG("misaki", "address", success) != "") {
        supportedComponents->support[MISAKI] = true;
    }
    if(GET_STRING_CONFIG("azuki", "address", success) != "") {
        supportedComponents->support[AZUKI] = true;
    }
    if(GET_STRING_CONFIG("shiori", "address", success) != "") {
        supportedComponents->support[SHIORI] = true;
    }
    if(GET_STRING_CONFIG("nozomi", "address", success) != "") {
        supportedComponents->support[NOZOMI] = true;
    }
    if(GET_STRING_CONFIG("inori", "address", success) != "") {
        supportedComponents->support[INORI] = true;
    }
}

/**
 * @brief add new server
 *
 * @param serverAddress address of the new server
 * @param error reference for error-output
 * @param port of the tcp-server
 * @param certFilePath path to the certificate-file
 * @param keyFilePath path to the key-file
 *
 * @return true, if successful, else false
 */
bool
HanamiMessaging::addServer(const std::string &serverAddress,
                           ErrorContainer &error,
                           const uint16_t port,
                           const std::string &certFilePath,
                           const std::string &keyFilePath)
{
    // init server based on the type of the address in the config
    const std::regex ipv4Regex("\\b((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\\.|$)){4}\\b");
    if(regex_match(serverAddress, ipv4Regex))
    {
        // create tcp-server
        if(m_sessionController->addTlsTcpServer(port, certFilePath, keyFilePath, error) == 0)
        {
            error.addMeesage("can't initialize tcp-server on port "
                             + std::to_string(port));
            LOG_ERROR(error);
            return false;
        }
    }
    else
    {
        // create uds-server
        if(m_sessionController->addUnixDomainServer(serverAddress, error) == 0)
        {
            error.addMeesage("can't initialize uds-server on file " + serverAddress);
            LOG_ERROR(error);
            return false;
        }
    }

    return true;
}

/**
 * @brief HanamiMessaging::createTemporaryClient
 * @param remoteIdentifier
 * @param error
 * @return
 */
HanamiMessagingClient*
HanamiMessaging::createTemporaryClient(const std::string &remoteIdentifier,
                                       const std::string &target,
                                       ErrorContainer &error)
{
    bool success = false;
    const std::string address = GET_STRING_CONFIG(target, "address", success);
    if(address != "")
    {
        const uint16_t port = static_cast<uint16_t>(GET_INT_CONFIG(target, "port", success));
        HanamiMessagingClient* newClient = new HanamiMessagingClient(remoteIdentifier,
                                                                     address,
                                                                     port);
        if(newClient->connectClient(error) == false)
        {
            delete newClient;
            return nullptr;
        }

        return newClient;
    }

    return nullptr;
}

/**
 * @brief initalize client-connections
 *
 * @param configGroups list of groups in config-file
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
HanamiMessaging::initClients(const std::vector<std::string> &configGroups,
                             ErrorContainer &error)
{
    bool success = false;

    // init client-connections
    for(const std::string& groupName : configGroups)
    {
        const std::string address = GET_STRING_CONFIG(groupName, "address", success);
        if(address != "")
        {
            const uint16_t port = static_cast<uint16_t>(GET_INT_CONFIG(groupName, "port", success));
            HanamiMessagingClient* newClient = new HanamiMessagingClient(groupName, address, port);
            newClient->startThread();
            m_clients.emplace(groupName, newClient);

            if(groupName == "misaki") {
                misakiClient = newClient;
            }
            if(groupName == "shiori") {
                shioriClient = newClient;
            }
            if(groupName == "kyouko") {
                kyoukoClient = newClient;
            }
            if(groupName == "azuki") {
                azukiClient = newClient;
            }
            if(groupName == "nozomi") {
                nozomiClient = newClient;
            }
            if(groupName == "inori") {
                inoriClient = newClient;
            }
            if(groupName == "torii") {
                toriiClient = newClient;
            }
        }
    }

    // wait until all connected
    for(const auto& [name, client] : m_clients)
    {
        // TODO: make wait-time configurable
        if(client->waitForAllConnected(1) == false)
        {
            error.addMeesage("Failed to initalize connection for client '"
                             + name
                             + "'");
        }
    }

    return true;
}

/**
 * @brief create and initialize new messaging-controller
 *
 * @param localIdentifier identifier for outgoing sessions to identify against the servers
 * @param configGroups config-groups for automatic creation of server and clients
 * @param receiver receiver for handling of intneral steam-messages within the callback-function
 * @param processStream callback for stream-messages
 * @param processGenericRequest callback for data-request-messages
 * @param error callbacks for incoming stream-messages
 * @param createServer true, if the instance should also create a server
 * @param predefinedEndpoints
 *
 * @return true, if successful, else false
 */
bool
HanamiMessaging::initialize(const std::string &localIdentifier,
                            const std::vector<std::string> &configGroups,
                            void* receiver,
                            void (*processStream)(void*,
                                                  Sakura::Session*,
                                                  const void*,
                                                  const uint64_t),
                            void (*processGenericRequest)(Sakura::Session*,
                                                          const uint32_t,
                                                          void*,
                                                          const uint64_t,
                                                          const uint64_t),
                            ErrorContainer &error,
                            const bool createServer)
{
    // precheck to avoid double-initializing
    if(m_isInit) {
        return false;
    }

    // set callback to send error-messages to shiori for logging
    setErrorLogCallback(&handleErrorCallback);

    // init client-handler
    this->streamReceiver = receiver;
    this->processStreamData = processStream;
    this->processGenericRequest = processGenericRequest;

    // check if config-file already initialized
    if(ConfigHandler::m_config == nullptr)
    {
        error.addMeesage("config-file not initilized");
        LOG_ERROR(error);
        return false;
    }

    // init config-options
    registerBasicConnectionConfigs(configGroups, createServer, error);
    if(ConfigHandler::m_config->isConfigValid() == false) {
        return false;
    }
    fillSupportOverview();
    SupportedComponents* support = SupportedComponents::getInstance();
    support->localComponent = localIdentifier;

    // init server if requested
    if(createServer)
    {
        // get server-address from config
        bool success = false;
        const std::string serverAddress = GET_STRING_CONFIG("DEFAULT", "address", success);
        if(success == false)
        {
            error.addMeesage("Failed to get server-address from config.");
            LOG_ERROR(error);
            return false;
        }

        // get port from config
        const int port = GET_INT_CONFIG("DEFAULT", "port", success);
        const uint16_t serverPort = static_cast<uint16_t>(port);

        // create server
        if(addServer(serverAddress, error, serverPort) == false)
        {
            error.addMeesage("Failed to create server on address '"
                             + serverAddress
                             + "' with port '"
                             + std::to_string(serverPort)
                             + "'.");
            LOG_ERROR(error);
            return false;
        }
    }

    // init clients
    if(initClients(configGroups, error) == false)
    {
        error.addMeesage("Failed to initialize clients.");
        LOG_ERROR(error);
        return false;
    }

    m_isInit = true;

    return true;
}

/**
 * @brief close a client
 *
 * @param remoteIdentifier identifier for the client to close
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
HanamiMessaging::closeClient(const std::string &remoteIdentifier,
                             ErrorContainer &error)
{
    std::map<std::string, HanamiMessagingClient*>::iterator it;
    it = m_clients.find(remoteIdentifier);
    if(it != m_clients.end()) {
        return it->second->closeClient(error);
    }

    return false;
}

/**
 * @brief get outgoing client-pointer
 *
 * @param identifier target-identifier to specifiy the client
 *
 * @return pointer to client, if found, else nullptr
 */
HanamiMessagingClient*
HanamiMessaging::getOutgoingClient(const std::string &identifier)
{
    std::map<std::string, HanamiMessagingClient*>::iterator it;
    it = m_clients.find(identifier);
    if(it != m_clients.end()) {
        return it->second;
    }

    return nullptr;
}

/**
 * @brief get instance, which must be already initialized
 *
 * @return instance-object
 */
HanamiMessaging*
HanamiMessaging::getInstance()
{
    if(m_messagingController == nullptr) {
        m_messagingController = new HanamiMessaging();
    }
    return m_messagingController;
}

/**
 * @brief send error-message to shiori
 *
 * @param errorMessage error-message to send to shiori
 */
void
HanamiMessaging::sendGenericErrorMessage(const std::string &errorMessage)
{
    // check if shiori is supported
    if(SupportedComponents::getInstance()->support[SHIORI] == false) {
        return;
    }

    // this function is triggered by every error-message in this logger. if an error is in the
    // following code which sending to shiori, it would result in an infinity-look of this function
    // and a stackoverflow. So this variable should ensure, that error in this function doesn't
    // tigger the function in itself again.
    if(m_whileSendError == true) {
        return;
    }
    m_whileSendError = true;

    // create message
    HanamiMessagingClient* client = HanamiMessaging::getInstance()->shioriClient;
    if(client == nullptr) {
        return;
    }

    // create binary for send
    ErrorLog_Message msg;
    msg.set_errormsg(errorMessage);

    // serialize message
    const uint64_t msgSize = msg.ByteSizeLong();
    uint8_t* buffer = new uint8_t[msgSize];
    if(msg.SerializeToArray(buffer, msgSize) == false)
    {
        m_whileSendError = false;
        return;
    }

    // send message
    Kitsunemimi::ErrorContainer error;
    const bool ret = client->sendGenericMessage(SHIORI_ERROR_LOG_MESSAGE_TYPE,
                                                buffer,
                                                msgSize,
                                                error);
    delete[] buffer;
    if(ret == false)
    {
        m_whileSendError = false;
        return;
    }

    m_whileSendError = false;
}

/**
 * @brief register an incoming connection
 *
 * @param identifier identifier for the new incoming connection
 * @param newSession pointer to the session
 *
 * @return true, if successful, else false
 */
bool
HanamiMessaging::addInternalClient(const std::string &identifier,
                                   Sakura::Session* newSession)
{
    m_incominglock.lock();

    // check if client-identifier is already registered
    std::map<std::string, HanamiMessagingClient*>::const_iterator it;
    it = m_incomingClients.find(identifier);
    if(it != m_incomingClients.end())
    {
        m_incominglock.unlock();

        ErrorContainer error;
        newSession->closeSession(error);
        LOG_ERROR(error);
        // TODO: handle deletion of internal connection
        return false;
    }

    // register client
    HanamiMessagingClient* newInternalCient = new HanamiMessagingClient(identifier, "", 0);
    newInternalCient->replaceSession(newSession);
    m_incomingClients.insert(std::make_pair(identifier, newInternalCient));

    m_incominglock.unlock();

    return true;
}

/**
 * @brief request the session of an incoming connection
 *
 * @param identifier identifier of the connection
 *
 * @return nullptr, if session for the identifier was not found, else pointer to the found session
 */
HanamiMessagingClient*
HanamiMessaging::getIncomingClient(const std::string &identifier)
{
    std::lock_guard<std::mutex> guard(m_incominglock);

    std::map<std::string, HanamiMessagingClient*>::const_iterator it;
    it = m_incomingClients.find(identifier);
    if(it != m_incomingClients.end()) {
        return it->second;
    }

    return nullptr;
}

/**
 * @brief remove the client of an incoming connection
 *
 * @param identifier identifier for the internal client
 *
 * @return true, if successful, else false
 */
bool
HanamiMessaging::removeInternalClient(const std::string &identifier)
{
    m_incominglock.lock();

    std::map<std::string, HanamiMessagingClient*>::iterator it;
    it = m_incomingClients.find(identifier);
    if(it != m_incomingClients.end())
    {
        HanamiMessagingClient* tempSession = it->second;
        m_incomingClients.erase(it);

        m_incominglock.unlock();

        if(tempSession != nullptr)
        {
            ErrorContainer error;
            if(tempSession->closeClient(error) == false) {
                LOG_ERROR(error);
            }

            delete tempSession;
        }

        return true;
    }

    m_incominglock.unlock();

    return false;
}


/**
 * @brief check if a specific blossom was registered
 *
 * @param groupName group-identifier of the blossom
 * @param itemName item-identifier of the blossom
 *
 * @return true, if blossom with the given group- and item-name exist, else false
 */
bool
HanamiMessaging::doesBlossomExist(const std::string &groupName,
                                  const std::string &itemName)
{
    std::map<std::string, std::map<std::string, Blossom*>>::const_iterator groupIt;
    groupIt = m_registeredBlossoms.find(groupName);

    if(groupIt != m_registeredBlossoms.end())
    {
        std::map<std::string, Blossom*>::const_iterator itemIt;
        itemIt = groupIt->second.find(itemName);

        if(itemIt != groupIt->second.end()) {
            return true;
        }
    }

    return false;
}

/**
 * @brief SakuraLangInterface::addBlossom
 *
 * @param groupName group-identifier of the blossom
 * @param itemName item-identifier of the blossom
 * @param newBlossom pointer to the new blossom
 *
 * @return true, if blossom was registered or false, if the group- and item-name are already
 *         registered
 */
bool
HanamiMessaging::addBlossom(const std::string &groupName,
                            const std::string &itemName,
                            Blossom* newBlossom)
{
    // check if already used
    if(doesBlossomExist(groupName, itemName) == true) {
        return false;
    }

    std::map<std::string, std::map<std::string, Blossom*>>::iterator groupIt;
    groupIt = m_registeredBlossoms.find(groupName);

    // create internal group-map, if not already exist
    if(groupIt == m_registeredBlossoms.end())
    {
        std::map<std::string, Blossom*> newMap;
        m_registeredBlossoms.insert(std::make_pair(groupName, newMap));
    }

    // add item to group
    groupIt = m_registeredBlossoms.find(groupName);
    groupIt->second.insert(std::make_pair(itemName, newBlossom));

    return true;
}

/**
 * @brief request a registered blossom
 *
 * @param groupName group-identifier of the blossom
 * @param itemName item-identifier of the blossom
 *
 * @return pointer to the blossom or
 *         nullptr, if blossom the given group- and item-name was not found
 */
Blossom*
HanamiMessaging::getBlossom(const std::string &groupName,
                            const std::string &itemName)
{
    // search for group
    std::map<std::string, std::map<std::string, Blossom*>>::const_iterator groupIt;
    groupIt = m_registeredBlossoms.find(groupName);

    if(groupIt != m_registeredBlossoms.end())
    {
        // search for item within group
        std::map<std::string, Blossom*>::const_iterator itemIt;
        itemIt = groupIt->second.find(itemName);

        if(itemIt != groupIt->second.end()) {
            return itemIt->second;
        }
    }

    return nullptr;
}

/**
 * @brief trigger existing blossom
 *
 * @param result map with resulting items
 * @param blossomName id of the blossom to trigger
 * @param blossomGroupName id of the group of the blossom to trigger
 * @param initialValues input-values for the tree
 * @param status reference for status-output
 * @param error reference for error-output
 *
 * @return true, if successfule, else false
 */
bool
HanamiMessaging::triggerBlossom(DataMap &result,
                                const std::string &blossomName,
                                const std::string &blossomGroupName,
                                const DataMap &context,
                                const DataMap &initialValues,
                                BlossomStatus &status,
                                ErrorContainer &error)
{
    LOG_DEBUG("trigger blossom");

    // get initial blossom-item
    Blossom* blossom = getBlossom(blossomGroupName, blossomName);
    if(blossom == nullptr)
    {
        error.addMeesage("No blosom found for the id " + blossomName);
        return false;
    }

    // inialize a new blossom-leaf for processing
    BlossomIO blossomIO;
    blossomIO.blossomName = blossomName;
    blossomIO.blossomPath = blossomName;
    blossomIO.blossomGroupType = blossomGroupName;
    blossomIO.input = &initialValues;
    blossomIO.parentValues = blossomIO.input.getItemContent()->toMap();
    blossomIO.nameHirarchie.push_back("BLOSSOM: " + blossomName);

    std::string errorMessage;
    // check input to be complete
    if(blossom->validateFieldsCompleteness(initialValues,
                                           *blossom->getInputValidationMap(),
                                           FieldDef::INPUT_TYPE,
                                           errorMessage) == false)
    {
        error.addMeesage(errorMessage);
        error.addMeesage("check of completeness of input-fields failed");
        status.statusCode = 400;
        status.errorMessage = errorMessage;
        LOG_ERROR(error);
        return false;
    }

    // process blossom
    if(blossom->growBlossom(blossomIO, &context, status, error) == false)
    {
        error.addMeesage("trigger blossom failed.");
        LOG_ERROR(error);
        return false;
    }

    // check output to be complete
    DataMap* output = blossomIO.output.getItemContent()->toMap();
    if(blossom->validateFieldsCompleteness(*output,
                                           *blossom->getOutputValidationMap(),
                                           FieldDef::OUTPUT_TYPE,
                                           errorMessage) == false)
    {
        error.addMeesage(errorMessage);
        error.addMeesage("check of completeness of output-fields failed");
        status.statusCode = 500;
        status.errorMessage = errorMessage;
        LOG_ERROR(error);
        return false;
    }

    // TODO: override only with the output-values to avoid unnecessary conflicts
    result.clear();
    overrideItems(result, *output, ALL);

    return true;
}

/**
 * @brief map the endpoint to the real target
 *
 * @param result reference to the result to identify the target
 * @param id request-id
 * @param type requested http-request-type
 *
 * @return false, if mapping failes, else true
 */
bool
HanamiMessaging::mapEndpoint(EndpointEntry &result,
                             const std::string &id,
                             const HttpRequestType type)
{
    std::map<std::string, std::map<HttpRequestType, EndpointEntry>>::const_iterator id_it;
    id_it = endpointRules.find(id);

    if(id_it != endpointRules.end())
    {
        std::map<HttpRequestType, EndpointEntry>::const_iterator type_it;
        type_it = id_it->second.find(type);

        if(type_it != id_it->second.end())
        {
            result.type = type_it->second.type;
            result.group = type_it->second.group;
            result.name = type_it->second.name;
            return true;
        }
    }

    return false;
}

/**
 * @brief add new custom-endpoint without the parser
 *
 * @param id identifier for the new entry
 * @param httpType http-type (get, post, put, delete)
 * @param sakuraType sakura-type (tree or blossom)
 * @param group blossom-group
 * @param name tree- or blossom-id
 *
 * @return false, if id together with http-type is already registered, else true
 */
bool
HanamiMessaging::addEndpoint(const std::string &id,
                             const HttpRequestType &httpType,
                             const SakuraObjectType &sakuraType,
                             const std::string &group,
                             const std::string &name)
{
    EndpointEntry newEntry;
    newEntry.type = sakuraType;
    newEntry.group = group;
    newEntry.name = name;

    // search for id
    std::map<std::string, std::map<HttpRequestType, EndpointEntry>>::iterator id_it;
    id_it = endpointRules.find(id);
    if(id_it != endpointRules.end())
    {
        // search for http-type
        std::map<HttpRequestType, EndpointEntry>::iterator type_it;
        type_it = id_it->second.find(httpType);
        if(type_it != id_it->second.end()) {
            return false;
        }

        // add new
        id_it->second.emplace(httpType, newEntry);
    }
    else
    {
        // add new
        std::map<HttpRequestType, EndpointEntry> typeEntry;
        typeEntry.emplace(httpType, newEntry);
        endpointRules.emplace(id, typeEntry);
    }

    return true;
}

}
