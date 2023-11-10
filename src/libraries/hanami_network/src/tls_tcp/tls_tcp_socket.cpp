/**
 *  @file    tls_tcp_socket.cpp
 *
 *  @author  Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright MIT License
 */

#include <hanami_common/logger.h>
#include <hanami_common/threading/cleanup_thread.h>
#include <tls_tcp/tls_tcp_socket.h>

namespace Hanami
{

/**
 * @brief constructor for the socket-side of the tcp-connection
 *
 * @param address ipv4-adress of the server
 * @param port port where the server is listen
 * @param threadName thread-name
 * @param certFile path to certificate-file
 * @param keyFile path to key-file
 * @param caFile path to ca-file
 */
TlsTcpSocket::TlsTcpSocket(TcpSocket&& socket,
                           const std::string& certFile,
                           const std::string& keyFile,
                           const std::string& caFile)
{
    this->socket = std::move(socket);
    this->certFile = certFile;
    this->keyFile = keyFile;
    this->caFile = caFile;
}

/**
 * @brief default-constructor
 */
TlsTcpSocket::TlsTcpSocket() {}

/**
 * @brief destructor
 */
TlsTcpSocket::~TlsTcpSocket() { cleanupOpenssl(); }

/**
 * @brief get file-descriptor
 *
 * @return file-descriptor
 */
int
TlsTcpSocket::getSocketFd() const
{
    return socket.getSocketFd();
}

/**
 * @brief init socket on client-side
 *
 * @param error reference for error-output
 *
 * @return true, if successful, else false
 */
bool
TlsTcpSocket::initClientSide(ErrorContainer& error)
{
    if (socket.socketFd == 0) {
        if (socket.initClientSide(error) == false) {
            return false;
        }
    }

    if (initOpenssl(error) == false) {
        return false;
    }

    LOG_INFO("Successfully initialized encrypted tcp-socket client to targe: "
             + socket.getAddress());

    return true;
}

/**
 * @brief init ssl and bind it to the file-descriptor
 *
 * @param error reference for error-output
 */
bool
TlsTcpSocket::initOpenssl(ErrorContainer& error)
{
    int result = 0;

    // common ssl-init
    SSL_load_error_strings();
    OpenSSL_add_ssl_algorithms();
    SSL_library_init();

    // set ssl-type
    const SSL_METHOD* method;
    if (isClientSide()) {
        method = TLS_client_method();
    }
    else {
        method = TLS_server_method();
    }

    // init ssl-context
    m_ctx = SSL_CTX_new(method);
    if (m_ctx == nullptr) {
        error.addMeesage("Failed to create ssl-context object");
        return false;
    }
    SSL_CTX_set_options(m_ctx, SSL_OP_SINGLE_DH_USE);

    // set certificate
    result = SSL_CTX_use_certificate_file(m_ctx, certFile.c_str(), SSL_FILETYPE_PEM);
    if (result <= 0) {
        error.addMeesage("Failed to load certificate file for ssl-encrytion. File path: \""
                         + certFile + "\"");
        error.addSolution("check if file \"" + certFile+ "\" exist and "
                           "contains a valid certificate");
        return false;
    }

    // set key
    result = SSL_CTX_use_PrivateKey_file(m_ctx, keyFile.c_str(), SSL_FILETYPE_PEM);
    if (result <= 0) {
        error.addMeesage("Failed to load key file for ssl-encrytion. File path: " + keyFile);
        error.addSolution("check if file \"" + keyFile + "\" exist and contains a valid key");
        return false;
    }

    // set CA-file if exist
    if (caFile != "") {
        result = SSL_CTX_load_verify_locations(m_ctx, caFile.c_str(), nullptr);
        if (result <= 0) {
            error.addMeesage("Failed to load CA file for ssl-encrytion. File path: " + caFile);
            error.addSolution("check if file \"" + caFile + "\" exist and contains a valid CA");
            return false;
        }

        // set the verification depth to 1
        SSL_CTX_set_verify_depth(m_ctx, 1);
    }

    // init ssl-cennection
    m_ssl = SSL_new(m_ctx);
    if (m_ssl == nullptr) {
        error.addMeesage("Failed to ini ssl");
        return false;
    }
    SSL_set_fd(m_ssl, socket.socketFd);

    // enable certificate validation, if ca-file was set
    if (caFile != "") {
        // SSL_VERIFY_PEER -> check cert if exist
        // SSL_VERIFY_FAIL_IF_NO_PEER_CERT -> server requires cert
        // SSL_VERIFY_CLIENT_ONCE -> check only on initial handshake
        SSL_set_verify(m_ssl,
                       SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT | SSL_VERIFY_CLIENT_ONCE,
                       nullptr);
    }

    // process tls-handshake
    if (isClientSide()) {
        // try to connect to server
        result = SSL_connect(m_ssl);
        if (result <= 0) {
            error.addMeesage("Failed to perform ssl-handshake (client-side)");
            error.addSolution("Maybe the server is only plain TCP-server or doesn't support TLS");
            return false;
        }
    }
    else {
        // try to accept incoming ssl-connection
        int result = SSL_accept(m_ssl);
        if (result <= 0) {
            error.addMeesage("Failed to perform ssl-handshake (client-side)");
            error.addSolution(
                "Maybe the incoming client is only plain TCP-client "
                "or doesn't support TLS");
            return false;
        }
    }

    return true;
}

/**
 * @brief check if socket is on client-side of the connection
 *
 * @return true, if socket is client-side, else false
 */
bool
TlsTcpSocket::isClientSide() const
{
    return socket.isClientSide();
}

/**
 * @brief receive data
 *
 * @return number of read bytes
 */
long
TlsTcpSocket::recvData(int, void* bufferPosition, const size_t bufferSize, int)
{
    return SSL_read(m_ssl, bufferPosition, static_cast<int>(bufferSize));
}

/**
 * @brief send data
 *
 * @return number of written bytes
 */
ssize_t
TlsTcpSocket::sendData(int, const void* bufferPosition, const size_t bufferSize, int)
{
    return SSL_write(m_ssl, bufferPosition, static_cast<int>(bufferSize));
}

/**
 * @brief cleanup openssl
 */
bool
TlsTcpSocket::cleanupOpenssl()
{
    if (m_ssl != nullptr) {
        SSL_shutdown(m_ssl);
        SSL_free(m_ssl);
    }

    if (m_ctx == nullptr) {
        SSL_CTX_free(m_ctx);
    }

    ERR_free_strings();
    EVP_cleanup();

    return true;
}

}  // namespace Hanami
