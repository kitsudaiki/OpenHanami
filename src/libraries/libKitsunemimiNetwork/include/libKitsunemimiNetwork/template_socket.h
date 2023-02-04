/**
 *  @file    template_socket.h
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#ifndef KITSUNEMIMI_NETWORK_TEMPLATE_SOCKET_H
#define KITSUNEMIMI_NETWORK_TEMPLATE_SOCKET_H

#include <sys/types.h>
#include <sys/socket.h>

#include <libKitsunemimiNetwork/abstract_socket.h>
#include <libKitsunemimiCommon/threading/thread.h>
#include <libKitsunemimiCommon/logger.h>
#include <libKitsunemimiCommon/buffer/ring_buffer.h>
#include <libKitsunemimiCommon/threading/cleanup_thread.h>

namespace Kitsunemimi
{
struct RingBuffer;

class TcpSocket;
class UnixDomainSocket;
class TlsTcpSocket;

template<class T>
class TemplateSocket
        : public AbstractSocket
{
public:
    /**
     * @brief constructor
     *
     * @param socketbase-socket-object
     * @param threadName name of the thread of the socket
     */
    TemplateSocket(T&& socket,
                   const std::string &threadName)
        : AbstractSocket(threadName)
    {
        m_socket = std::move(socket);
    }

    /**
     * @brief destructor
     */
    ~TemplateSocket()
    {
        closeSocket();
    }

    /**
     * @brief initialize new connection to a server
     *
     * @param error reference for error-output
     *
     * @return true, if successful, else false
     */
    bool initConnection(Kitsunemimi::ErrorContainer &error)
    {
        return m_socket.initClientSide(error);
    }

    /**
     * @brief get socket-type
     *
     * @return socket-type enum
     */
    uint32_t getType()
    {
        return m_socket.type;
    }

    /**
     * @brief check if socket is on client-side
     *
     * @return true, if socket is on client-side of the connection
     */
    bool isClientSide()
    {
        return m_socket.isClientSide();
    }

    /**
     * @brief send a text-message over the socket
     *
     * @param message message to send
     * @param error reference for error-output
     *
     * @return false, if send failed or send was incomplete, else true
     */
    bool sendMessage(const std::string &message, ErrorContainer &error)
    {
        const uint64_t messageLength = message.length();
        return sendMessage(static_cast<const void*>(message.c_str()), messageLength, error);
    }

    /**
     * @brief send a byte-buffer over the tcp-socket
     *
     * @param message byte-buffer to send
     * @param numberOfBytes number of bytes to send
     * @param error reference for error-output
     *
     * @return false, if send failed or send was incomplete, else true
     */
    bool sendMessage(const void* message,
                     const uint64_t numberOfBytes,
                     ErrorContainer &error)
    {
        // precheck if socket is connected
        if(m_socket.getSocketFd() == 0)
        {
            error.addMeesage("socket is not connected");
            return false;
        }

        // send message
        while(m_lock.test_and_set(std::memory_order_acquire)) { asm(""); }
        const ssize_t successfulSended = m_socket.sendData(m_socket.getSocketFd(),
                                                           message,
                                                           numberOfBytes,
                                                           MSG_NOSIGNAL);
        m_lock.clear(std::memory_order_release);

        // check if the message was completely send
        if(successfulSended < -1
                || successfulSended != static_cast<long>(numberOfBytes))
        {
            return false;
        }

        return true;
    }

    /**
     * @brief close the socket and schedule the deletion of the thread
     *
     * @return false, if already closed, else true
     */
    bool closeSocket()
    {
        if(m_abort == true) {
            return false;
        }

        m_abort = true;

        // close socket if connected
        if(m_socket.getSocketFd() >= 0)
        {
            shutdown(m_socket.getSocketFd(), SHUT_RDWR);
            close(m_socket.getSocketFd());
            //m_socket.getSocketFd() = 0;
        }

        // make sure, that the thread is out of the function recvData before further
        // deleteing the thread (maximum wait-time = 10ms)
        int32_t timeout = 100;
        while(m_isfullyClosed == false
              && timeout > 0)
        {
            usleep(100);
            timeout--;
        }

        return true;
    }

private:
    Kitsunemimi::RingBuffer m_recvBuffer;
    std::atomic_flag m_lock = ATOMIC_FLAG_INIT;
    T m_socket;

    /**
     * @brief wait for new incoming messages
     *
     * @return false, if receive failed or socket is aborted, else true
     */
    bool waitForMessage()
    {
        // precheck
        if(m_abort) {
            return true;
        }

        // calulate buffer-part for recv message
        const uint64_t writePosition = Kitsunemimi::getWritePosition_RingBuffer(m_recvBuffer);
        const uint64_t spaceToEnd = Kitsunemimi::getSpaceToEnd_RingBuffer(m_recvBuffer);

        // wait for incoming message
        const long recvSize = m_socket.recvData(m_socket.getSocketFd(),
                                                &m_recvBuffer.data[writePosition],
                                                spaceToEnd,
                                                0);

        // handle error-cases
        if(recvSize <= 0
                || m_abort)
        {
            return true;
        }

        // increase the
        m_recvBuffer.usedSize = (m_recvBuffer.usedSize + static_cast<uint64_t>(recvSize));

        // process message via callback-function
        uint64_t readBytes = 0;
        do
        {
            readBytes = m_processMessage(m_target, &m_recvBuffer, this);
            moveForward_RingBuffer(m_recvBuffer, readBytes);
        }
        while(readBytes > 0);

        return true;
    }

    bool m_isfullyClosed = false;

protected:
    /**
     * @brief run-method for the thread-class
     */
    void run()
    {
        while(m_abort == false) {
            waitForMessage();
        }
        m_isfullyClosed = true;
    }
};

} // namespace Kitsunemimi

#endif // KITSUNEMIMI_NETWORK_TEMPLATE_SOCKET_H
