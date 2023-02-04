/**
 *  @file       jwt.h
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

#ifndef KITSUNEMIMI_JWT_H
#define KITSUNEMIMI_JWT_H

#include <cryptopp/secblock.h>
#include <chrono>

#include <libKitsunemimiCommon/logger.h>

namespace Kitsunemimi
{
class JsonItem;

bool getJwtTokenPayload(JsonItem &parsedResult,
                        const std::string &token,
                        ErrorContainer &error);

class Jwt
{
public:
    Jwt(const CryptoPP::SecByteBlock &signingKey);

    bool create_HS256_Token(std::string &result,
                            JsonItem &payload,
                            const u_int32_t validSeconds,
                            ErrorContainer &error);

    bool validateToken(JsonItem &resultPayload,
                       const std::string &token,
                       std::string &publicError,
                       ErrorContainer &error);

private:
    CryptoPP::SecByteBlock m_signingKey;

    // signature
    bool validateSignature(const std::string &alg,
                           const std::string &relevantPart,
                           const std::string &signature,
                           ErrorContainer &error);
    bool validate_HS256_Signature(const std::string &relevantPart,
                                  const std::string &signature,
                                  ErrorContainer &error);

    // times
    void addTimesToPayload(JsonItem &payload,
                           const u_int32_t validSeconds);
    bool checkTimesInPayload(const JsonItem &payload,
                             ErrorContainer &error);
    long getTimeSinceEpoch();
};

}  // namespace Kitsunemimi

#endif // KITSUNEMIMI_JWT_H
