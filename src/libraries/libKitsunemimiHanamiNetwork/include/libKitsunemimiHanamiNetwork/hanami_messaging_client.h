#ifndef KITSUNEMIMI_HANAMI_NETWORK_HANAMIMESSAGINGCLIENT_H
#define KITSUNEMIMI_HANAMI_NETWORK_HANAMIMESSAGINGCLIENT_H

#include <iostream>
#include <map>
#include <vector>
#include <mutex>
#include <regex>

#include <libKitsunemimiHanamiCommon/enums.h>
#include <libKitsunemimiHanamiCommon/structs.h>

#include <libKitsunemimiCommon/logger.h>
#include <libKitsunemimiCommon/threading/thread.h>

namespace Kitsunemimi
{
struct DataBuffer;
class DataMap;
struct StackBuffer;
namespace Sakura {
class Blossom;
class Session;
}
namespace Hanami
{
class ClientHandler;
class HanamiMessaging;

class HanamiMessagingClient
        : public Kitsunemimi::Thread
{
public:
    bool sendStreamMessage(StackBuffer &data,
                           ErrorContainer &error);
    bool sendStreamMessage(const void* data,
                           const uint64_t dataSize,
                           const bool replyExpected,
                           ErrorContainer &error);

    bool sendGenericMessage(const uint32_t subType,
                            const void* data,
                            const uint64_t dataSize,
                            ErrorContainer &error);

    DataBuffer* sendGenericRequest(const uint32_t subType,
                                   const void* data,
                                   const uint64_t dataSize,
                                   ErrorContainer &error);

    bool triggerSakuraFile(ResponseMessage &response,
                           const RequestMessage &request,
                           ErrorContainer &error);

    bool setStreamCallback(void* receiver,
                           void (*processStream)(void*,
                                                 Sakura::Session*,
                                                 const void*,
                                                 const uint64_t));
    bool closeClient(ErrorContainer &error);
    bool connectClient(ErrorContainer &error);

protected:
    void run();

private:
    friend ClientHandler;
    friend HanamiMessaging;

    HanamiMessagingClient(const std::string &remoteIdentifier,
                          const std::string &address,
                          const uint16_t port);
    ~HanamiMessagingClient();


    std::string m_remoteIdentifier = "";
    std::string m_address = "";
    uint16_t m_port = 0;
    Sakura::Session* m_session = nullptr;
    std::mutex m_sessionLock;

    void replaceSession(Sakura::Session* newSession);
    bool waitForAllConnected(const uint32_t timeout);

    bool createRequest(Kitsunemimi::Sakura::Session* session,
                       ResponseMessage& response,
                       const RequestMessage &request,
                       ErrorContainer &error);
    bool processResponse(ResponseMessage& response,
                         const DataBuffer* responseData,
                         ErrorContainer &error);
};

}  // namespace Hanami
}  // namespace Kitsunemimi

#endif // HKITSUNEMIMI_HANAMI_NETWORK_ANAMIMESSAGINGCLIENT_H
