#include <libKitsunemimiHanamiNetwork/hanami_messaging_client.h>

#include <message_handling/message_definitions.h>

#include <libKitsunemimiHanamiNetwork/hanami_messaging.h>
#include <libKitsunemimiHanamiCommon/component_support.h>
#include <libKitsunemimiHanamiCommon/structs.h>
#include <libKitsunemimiHanamiCommon/functions.h>

#include <libKitsunemimiSakuraNetwork/session.h>
#include <libKitsunemimiSakuraNetwork/session_controller.h>

namespace Kitsunemimi::Hanami
{

/**
 * @brief private constructor
 *
 * @param remoteIdentifier indentifier with the name of the target
 * @param address target-address
 * @param port target-port
 */
HanamiMessagingClient::HanamiMessagingClient(const std::string &remoteIdentifier,
                                             const std::string &address,
                                             const uint16_t port)
    : Kitsunemimi::Thread("HanamiMessagingClient-" + remoteIdentifier)
{
    std::lock_guard<std::mutex> guard(m_sessionLock);

    m_remoteIdentifier = remoteIdentifier;
    m_address = address;
    m_port = port;
}

/**
 * @brief private constructor
 */
HanamiMessagingClient::~HanamiMessagingClient()
{
    ErrorContainer error;
    if(closeClient(error) == false) {
        LOG_ERROR(error);
    }
}

/**
 * @brief set callback for stram-message
 */
bool
HanamiMessagingClient::setStreamCallback(void* receiver,
                                         void (*processStream)(void*,
                                                               Sakura::Session*,
                                                               const void*,
                                                               const uint64_t))
{
    std::lock_guard<std::mutex> guard(m_sessionLock);

    if(m_session == nullptr) {
        return false;
    }

    m_session->setStreamCallback(receiver, processStream);
    return true;
}

/**
 * @brief close the session of the client
 *
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
HanamiMessagingClient::closeClient(ErrorContainer &error)
{
    std::lock_guard<std::mutex> guard(m_sessionLock);

    if(m_session == nullptr)
    {
        error.addMeesage("Hanami-client is not initialized with a session");
        return false;
    }

    if(m_session->closeSession(error) == false)
    {
        error.addMeesage("Closing Hanami-client failed");
        return false;
    }

    delete m_session;
    m_session = nullptr;

    return true;
}

/**
 * @brief send stream-message over a client
 *
 * @param data stack-buffer to send
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
HanamiMessagingClient::sendStreamMessage(StackBuffer &data,
                                         ErrorContainer &error)
{
    std::lock_guard<std::mutex> guard(m_sessionLock);

    if(m_session == nullptr)
    {
        error.addMeesage("Hanami-client is not initialized with a session");
        return false;
    }

    // send all data from the stackbuffer
    for(uint32_t i = 0; i < data.blocks.size(); i++)
    {
        DataBuffer* buf = getFirstElement_StackBuffer(data);

        // only the last buffer should have an expected reply
        const bool expReply = i == data.blocks.size() - 1;
        if(m_session->sendStreamData(buf->data, buf->usedBufferSize, error, expReply) == false) {
            return false;
        }
        removeFirst_StackBuffer(data);
    }

    return true;
}

/**
 * @brief send stream-message over a client
 *
 * @param data pointer to the data to send
 * @param dataSize size of data in bytes to send
 * @param replyExpected true to expect a reply-message
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
HanamiMessagingClient::sendStreamMessage(const void* data,
                                         const uint64_t dataSize,
                                         const bool replyExpected,
                                         ErrorContainer &error)
{
    std::lock_guard<std::mutex> guard(m_sessionLock);

    if(m_session == nullptr)
    {
        error.addMeesage("Hanami-client is not initialized with a session");
        return false;
    }

    return m_session->sendStreamData(data, dataSize, error, replyExpected);
}

/**
 * @brief send a generic message over the internal messaging
 *
 * @param subType message-subtype for identifiacation of the correct package
 * @param data pointer to data to send
 * @param dataSize size of data to send
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
HanamiMessagingClient::sendGenericMessage(const uint32_t subType,
                                          const void* data,
                                          const uint64_t dataSize,
                                          ErrorContainer &error)
{
    std::lock_guard<std::mutex> guard(m_sessionLock);

    // get client
    if(m_session == nullptr)
    {
        error.addMeesage("Hanami-client is not initialized with a session");
        return false;
    }

    // create header
    SakuraGenericHeader header;
    header.size = dataSize;
    header.subType = subType;

    // create message
    const uint64_t bufferSize = sizeof(SakuraGenericHeader) + dataSize;
    uint8_t* buffer = new uint8_t[bufferSize];
    memcpy(&buffer[0], &header, sizeof(SakuraGenericHeader));
    memcpy(&buffer[sizeof(SakuraGenericHeader)], data, dataSize);

    // send
    const bool ret = m_session->sendNormalMessage(buffer, bufferSize, error);
    delete[] buffer;

    return ret;
}

/**
 * @brief send a generic message over the internal messaging
 *
 * @param subType message-subtype for identifiacation of the correct package
 * @param data pointer to data to send
 * @param dataSize size of data to send
 * @param error reference for error-output
 *
 * @return pointer to data-buffer with response, if successful, else nullptr
 */
DataBuffer*
HanamiMessagingClient::sendGenericRequest(const uint32_t subType,
                                          const void* data,
                                          const uint64_t dataSize,
                                          ErrorContainer &error)
{
    std::lock_guard<std::mutex> guard(m_sessionLock);

    // get client
    if(m_session == nullptr)
    {
        error.addMeesage("Hanami-client is not initialized with a session");
        return nullptr;
    }

    // create header
    SakuraGenericHeader header;
    header.size = dataSize;
    header.subType = subType;

    // create message
    const uint64_t bufferSize = sizeof(SakuraGenericHeader) + dataSize;
    uint8_t* buffer = new uint8_t[bufferSize];
    memcpy(&buffer[0], &header, sizeof(SakuraGenericHeader));
    memcpy(&buffer[sizeof(SakuraGenericHeader)], data, dataSize);

    // send
    DataBuffer* result = m_session->sendRequest(buffer, bufferSize, 10, error);
    delete[] buffer;

    return result;
}

/**
 * @brief trigger remote action
 *
 * @param target name of the client to trigger
 * @param response reference for the response
 * @param request request-information to identify the target-action on the remote host
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
HanamiMessagingClient::triggerSakuraFile(ResponseMessage& response,
                                         const RequestMessage &request,
                                         ErrorContainer &error)
{
    std::lock_guard<std::mutex> guard(m_sessionLock);

    // get client
    if(m_session == nullptr)
    {
        error.addMeesage("Hanami-client is not initialized with a session");
        return false;
    }

    // try to send request to target
    if(createRequest(m_session, response, request, error) == false)
    {
        response.success = false;
        response.type = INTERNAL_SERVER_ERROR_RTYPE;
        error.addMeesage("Failed to trigger sakura-file.");
        return false;
    }

    return true;
}

/**
 * @brief HanamiMessagingClient::replaceSession
 * @param newSession
 */
void
HanamiMessagingClient::replaceSession(Sakura::Session* newSession)
{
    std::lock_guard<std::mutex> guard(m_sessionLock);

    m_session = newSession;
}

/**
 * @brief create a new connection to a client
 *
 * @param error reference for error-ourput
 *
 * @return true, if successful, else false
 */
bool
HanamiMessagingClient::connectClient(ErrorContainer &error)
{
    LOG_DEBUG("create client with remote-identifier \""
              + m_remoteIdentifier
              + "\" and address\""
              + m_address
              + "\"");

    Kitsunemimi::Sakura::Session* newSession = nullptr;
    Kitsunemimi::Sakura::SessionController* sessionCon = HanamiMessaging::m_sessionController;
    std::string localIdent = SupportedComponents::getInstance()->localComponent;
    if(isUuid(m_remoteIdentifier)) {
        localIdent = m_remoteIdentifier;
    }

    // connect based on the address-type
    const std::regex ipv4Regex(IPV4_REGEX);
    if(regex_match(m_address, ipv4Regex))
    {
        newSession = sessionCon->startTcpSession(m_address,
                                                 m_port,
                                                 localIdent,
                                                 "HanamiClient",
                                                 error);
    }
    else
    {
        newSession = sessionCon->startUnixDomainSession(m_address,
                                                        localIdent,
                                                        "HanamiClient",
                                                        error);
    }

    // check if connection was successful
    if(newSession == nullptr)
    {
        error.addMeesage("Failed to initialize session to address '" + m_address + "'");
        return false;
    }

    // handle result
    newSession->m_sessionIdentifier = m_remoteIdentifier;
    replaceSession(newSession);

    return true;
}

/**
 * @brief thread to close and delete connections, which are marked by a close-call or timeout,
 *        and to reconnect outgoing connections
 */
void
HanamiMessagingClient::run()
{
    while(m_abort == false)
    {
        if(m_session == nullptr)
        {
            ErrorContainer error;
            if(connectClient(error) == false)
            {
                error.addMeesage("create connection to '"
                                 + m_remoteIdentifier
                                 + "' failed");
                error.addSolution("check if component '"
                                  + m_remoteIdentifier
                                  + "' is up and running.");
                LOG_ERROR(error);
            }
        }

        // wait for 100ms
        sleepThread(100000);
    }
}

/**
 * @brief wait until all outging connections are connected
 *
 * @param timeout number of seconds to wait for a timeout
 *
 * @return true, if all are connected, else false
 */
bool
HanamiMessagingClient::waitForAllConnected(const uint32_t timeout)
{
    const uint64_t microTimeout = timeout * 1000000;
    uint64_t actualTimeout = 0;

    while(m_abort == false
            && actualTimeout < microTimeout)
    {
        if(m_session != nullptr) {
            return true;
        }

        sleepThread(10000);
        actualTimeout += 10000;
    }

    return false;
}

/**
 * @brief process response-message
 *
 * @param result reference for resulting data-items, which are withing the response
 * @param response data-buffer with the plain response message
 * @param errorMessage reference for error-output
 *
 * @return false, if message is invalid or process was not successful, else true
 */
bool
HanamiMessagingClient::processResponse(ResponseMessage& response,
                                       const DataBuffer* responseData,
                                       ErrorContainer &error)
{
    // precheck
    if(responseData->usedBufferSize == 0
            || responseData->data == nullptr)
    {
        error.addMeesage("missing message-content");
        LOG_ERROR(error);
        return false;
    }

    // transform incoming message
    const ResponseHeader* header = static_cast<const ResponseHeader*>(responseData->data);
    const char* message = static_cast<const char*>(responseData->data);
    const uint32_t pos = sizeof (ResponseHeader);
    const std::string messageContent(&message[pos], header->messageSize);
    response.success = header->success;
    response.type = header->responseType;
    response.responseContent = messageContent;

    LOG_DEBUG("received message with content: \'" + response.responseContent + "\'");

    return true;
}

/**
 * @brief trigger sakura-file remotely
 *
 * @param result resulting data-items coming from the triggered tree
 * @param id tree-id to trigger
 * @param inputValues input-values as string
 * @param errorMessage reference for error-output
 *
 * @return true, if successful, else false
 */
bool
HanamiMessagingClient::createRequest(Kitsunemimi::Sakura::Session* session,
                                     ResponseMessage& response,
                                     const RequestMessage &request,
                                     ErrorContainer &error)
{
    // create buffer
    const uint64_t totalSize = sizeof(SakuraTriggerHeader)
                               + request.id.size()
                               + request.inputValues.size();
    uint8_t* buffer = new uint8_t[totalSize];
    uint32_t positionCounter = 0;

    // prepare header
    SakuraTriggerHeader header;
    header.idSize = static_cast<uint32_t>(request.id.size());
    header.requestType = request.httpType;
    header.inputValuesSize = static_cast<uint32_t>(request.inputValues.size());

    // copy header
    memcpy(buffer, &header, sizeof(SakuraTriggerHeader));
    positionCounter += sizeof(SakuraTriggerHeader);

    // copy id
    memcpy(buffer + positionCounter, request.id.c_str(), request.id.size());
    positionCounter += request.id.size();

    // copy input-values
    memcpy(buffer + positionCounter, request.inputValues.c_str(), request.inputValues.size());

    // send
    // TODO: make timeout-time configurable
    DataBuffer* responseData = session->sendRequest(buffer, totalSize, 0, error);
    delete[] buffer;
    if(responseData == nullptr)
    {
        error.addMeesage("Timeout while triggering sakura-file with id: " + request.id);
        LOG_ERROR(error);
        return false;
    }

    const bool ret = processResponse(response, responseData, error);
    delete responseData;

    return ret;
}

}
