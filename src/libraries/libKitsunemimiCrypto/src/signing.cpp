/**
 *  @file       signing.cpp
 *
 *  @author     Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 *  @copyright  Apache License Version 2.0
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

#include <libKitsunemimiCrypto/signing.h>
#include <libKitsunemimiCrypto/common.h>

namespace Kitsunemimi
{

// IMPORTANT: Workaround to use the HMAC-functions unter OpenSSL 3, avoided update for now to
// avoid compatibility problems with older version. Will update to the new EVP_MAC-functions later
#undef OPENSSL_API_COMPAT
#define OPENSSL_API_COMPAT 0x10101000L

/**
 * @brief create a base64 encoded HMAC-value from an input-string
 *
 * @param result reference for the resulting string
 * @param input input-string to create the HMAC-value
 * @param key key for creating the HMAC
 * @param error reference for error-output
 *
 * @return false, if input is invalid, else true
 */
bool
create_HMAC_SHA256(std::string &result,
                   const std::string &input,
                   const CryptoPP::SecByteBlock &key,
                   ErrorContainer &error)
{
    if(input.size() == 0)
    {
        error.addMeesage("Creating HMAC failed, because input is empty");
        return false;
    }
    if(key.size() == 0)
    {
        error.addMeesage("Creating HMAC failed, because key is empty");
        return false;
    }

    unsigned int len = 32;
    unsigned char hmacResult[len];

    // calculate hmac signature
    HMAC_CTX* ctx = HMAC_CTX_new();
    HMAC_Init_ex(ctx,
                 key.data(),
                 key.size(),
                 EVP_sha256(),
                 nullptr);
    HMAC_Update(ctx, (unsigned char*)input.c_str(), input.length());
    HMAC_Final(ctx, hmacResult, &len);
    HMAC_CTX_free(ctx);

    result = std::string((char*)hmacResult, len);
    Kitsunemimi::encodeBase64(result, result.c_str(), sizeof(result));

    return true;
}

/**
 * @brief verify a HMAC-SHA256 input
 *
 * @param input input-string to verify
 * @param hmac base64-encoded HMAC-string, which belongs to the input
 * @param key key for verify
 *
 * @return true, if verification was successful, else false
 */
bool
verify_HMAC_SHA256(const std::string &input,
                   const std::string &hmac,
                   const CryptoPP::SecByteBlock &key)
{
    std::string compareHmac;

    // false in this function can also only mean, that the input doesn't match and doesn't have
    // to be an error-case, so the error-container is ony handled inter in this case
    ErrorContainer error;
    if(create_HMAC_SHA256(compareHmac, input, key, error) == false) {
        return false;
    }

    const bool result = hmac.size() == compareHmac.size()
                        && CRYPTO_memcmp(hmac.c_str(), compareHmac.c_str(), hmac.size()) == 0;

    return result;
}

}
