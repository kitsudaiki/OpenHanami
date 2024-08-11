# hanami_crypto

!!! warning

    This documentation here is the archived version of the old readme-file of this library and is NOT up-to-date, but maybe it is still useful for some references.

## Description

Wrapper-library for crypto-operation from other external libraries, to simplify the usage of basic
operation.

Actual support for:

-   base64 encode-decode
-   AES-256-CBC encryption
-   Sha256

## Usage

### Base64

Example to encode and decode base64-strings:

```cpp
#include <hanami_crypto/common.h>


// encode
std::string encodedStr = "";
const std::string input = "asdfasdfasdf123a";
Hanami::encodeBase64(encodedStr, input.c_str(), input.size());
// encodedStr has now the content: "YXNkZmFzZGZhc2RmMTIzYQ=="


// decode
std::string decodedStr;
Hanami::decodeBase64(decodedStr, encodedStr);
// decodedStr has now the content, which was the original input: "asdfasdfasdf123a"
// if the input was not a valid string, decodeBase64 return false
```

### AES-encryption

HINT: Actual only AES-CBC with 256-bit AES-key. Will be replaced by AES-XTS in the new future, but
XTS was not supported in the version of the crpyto++ library, which I had when creating this library
here.

Example for AES-CBC encyption and decryption:

```cpp
#include <hanami_crypto/symmetric_encryption.h>

Hanami::ErrorContainer error;

// demo-string to encrypt
const std::string testData = "this is a test-string";

// create a key for the encryption and decryption. The key doesn't need to have 256 bit length,
// because the encryption and decrytion uses internally a sha-function to bring the key to a
// valid length. So you can use here any string you want.
CryptoPP::SecByteBlock key((unsigned char*)"asdf", 4);

// encrypt
std::string encryptionResult;
Hanami::encrypt_AES_256_CBC(encryptionResult, testData, key, error);
// encryptionResult now contains the encrypted result

// decrypt
std::string decryptionResult;
Hanami::decrypt_AES_256_CBC(decryptionResult, encryptionResult, key, error);
```

Both functions return `false`, if something went wrong. In this case the error-message can be
converted to string with `error.toString()`.

### Sha256

Example for Sha256-hashing:

```cpp
#include <hanami_crypto/hashes.h>

std::string input = "test";
std::string result = "";
Hanami::generate_SHA_256(result, input);
// result now contains the string "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"


// the input doesn't have to be a string. You can also use a pointer to a byte-array like this
Hanami::generate_SHA_256(result, input.c_str(), input.size());
```
