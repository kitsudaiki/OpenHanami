#ifndef CERT_INIT_H
#define CERT_INIT_H

#include <iostream>
#include <libKitsunemimiCommon/files/text_file.h>

namespace Kitsunemimi
{
const std::string testCert = "-----BEGIN CERTIFICATE-----\n"
        "MIIDYDCCAkigAwIBAgIJAPrYys+kYk3BMA0GCSqGSIb3DQEBCwUAMEUxCzAJBgNV\n"
        "BAYTAkRFMRMwEQYDVQQIDApTb21lLVN0YXRlMSEwHwYDVQQKDBhJbnRlcm5ldCBX\n"
        "aWRnaXRzIFB0eSBMdGQwHhcNMTkwOTA1MDgzNzMwWhcNMjkwOTAyMDgzNzMwWjBF\n"
        "MQswCQYDVQQGEwJERTETMBEGA1UECAwKU29tZS1TdGF0ZTEhMB8GA1UECgwYSW50\n"
        "ZXJuZXQgV2lkZ2l0cyBQdHkgTHRkMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIB\n"
        "CgKCAQEA0+97FElGg2VIxIWlLJkjPfnW/1aE4HUzW3yrSftBNgpBTu7F4SNSoQo1\n"
        "h5KX5nHLXch56WwLzr32UDbMrlLCRovgkKCcLIz7Eg+3PR9ymi4snCtHzhojPh0l\n"
        "JMVBKL3x4onc+IcaybOcfi/FqjjFYndwad42xKF0MGm5bZ3iRnjAFwJyz7yJ3AFL\n"
        "6YoytCMf9de2NRSMkHyy9i31DCBHSiZHmaVTaxJI70BiMDbSsHkRvbzHP6mhYbxD\n"
        "P9aikNmJzFaZA2KWYzo9+G0GwoRxL9LU/17kefXJ8uu8Z2tsT2d0913H689TXG+b\n"
        "rVH58ikfZ0515+p+SB1gCY+G9CQ0XwIDAQABo1MwUTAdBgNVHQ4EFgQUGrq8glhx\n"
        "G+N5PZjOli1lINOZWsEwHwYDVR0jBBgwFoAUGrq8glhxG+N5PZjOli1lINOZWsEw\n"
        "DwYDVR0TAQH/BAUwAwEB/zANBgkqhkiG9w0BAQsFAAOCAQEAJuMkpiQDh+XZU3dS\n"
        "I4+Ji8xYkXnqahbSldHq5kXvVdcbknhU9roapjxEfxyygE9OH6+qdNEnQwKFcXOg\n"
        "EFKfydtOnGzapEyUXSEkK7Pw7roZ1LNspLcBhsNcBZQ8uR7F//hT/FPKiyDeqrYd\n"
        "NuDKZfBHFs/seLkOYMaSRGCG+LjY7EEpWaF2/o9yxMXyNn4gRtGMUeLZtPlIW8hx\n"
        "3cm9OWQDuuiodI5EONKJmWuFpwKWv4BEwcpXnwRb0Xvl2XKVKwRmnNWikvXurFWb\n"
        "+DT42R/MGefavgzlAFz46Ug1vYgfGCqdSBbJh/frm1WcANN1T2XoUnWug/QkheNu\n"
        "/DsexQ==\n"
        "-----END CERTIFICATE-----\n";

const std::string testKey = "-----BEGIN PRIVATE KEY-----\n"
        "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDT73sUSUaDZUjE\n"
        "haUsmSM9+db/VoTgdTNbfKtJ+0E2CkFO7sXhI1KhCjWHkpfmcctdyHnpbAvOvfZQ\n"
        "NsyuUsJGi+CQoJwsjPsSD7c9H3KaLiycK0fOGiM+HSUkxUEovfHiidz4hxrJs5x+\n"
        "L8WqOMVid3Bp3jbEoXQwabltneJGeMAXAnLPvIncAUvpijK0Ix/117Y1FIyQfLL2\n"
        "LfUMIEdKJkeZpVNrEkjvQGIwNtKweRG9vMc/qaFhvEM/1qKQ2YnMVpkDYpZjOj34\n"
        "bQbChHEv0tT/XuR59cny67xna2xPZ3T3Xcfrz1Ncb5utUfnyKR9nTnXn6n5IHWAJ\n"
        "j4b0JDRfAgMBAAECggEAPIpfbUcVRnmLVOAcc+X25EBXQy9S289+8TZms8Z7NVWu\n"
        "nD6m9g4iD3CcI/MjQyfkgRDAioZbxR4Mm5Nb2rw3VPGmH4pRsoQ/QESPAn3WPebM\n"
        "xXuzklNzF845iwxx9ZJ041KgdboaU93j6UP6Qgrfj6YwzX01xeudBitdVcvRFHHq\n"
        "r+jcIdZeL6rX9OFzsMqmTk8bM3gRVjeOKFvjlf8Dpsw2ziEofQStHKEFU+pW3MK3\n"
        "xLiwJFcQG7zomQGuNQPWql+NabpWaJpsyCzleon4qm6zjXI8vGJU3D5d7v4qhNDE\n"
        "zJDIzZwuH1NhOkt3adt81TM3dKdQhyFGuALt1zMg0QKBgQDzj7SIrC6TPzKi3uat\n"
        "iJUQqIFrFnxCh9J0UO1QrhPmJyIRcIO39gkKUUssBi2vQf7YiVW3opaY0OEwBGKN\n"
        "2YfONdhYdxSDcuBtzYsh7a3bRHViHewnao+q6M/gXdwhCnXMt6rjTPBzjVZ3cw3f\n"
        "l2fhWV5e2X62PWUS61wNjXFaOQKBgQDewk1hd3DPG0E687p4ypkigGtYXLph4NUt\n"
        "bjZKjhCfk7kvYxkQ4KbsXF+Lge0YVxYM9jRtHoqjVhwlvCdkOURrDlHXIK0xGwf6\n"
        "0t040GpLalBcy1plO++PZRUBlSJCp5Z3pSfOCeModF/0P4/uk/viAIlDPYhizbiC\n"
        "KEiy9I7jVwKBgQDnUsFAXWgO6aMKFXI5ttL8802Xi8+Q0LcNSh9a1TqJCPnOXnJ7\n"
        "se18IyGmOmgBYEjPGACVXJJzqU9273M7DjNxoqpLuy18ewq0vtc57ieFbUufWJQG\n"
        "C6tPw7ZLflmn9+tR988R+u0UklRhNqEijwZWfS6oHyG9rCnnAip3pLLX4QKBgFHf\n"
        "Cf+znXORKdVX9QYmOEg0+L8ePaZxswgihLO8KSHtcleXTYQlfVRL0xX8J78VatZS\n"
        "uwwL+Jp1sJyx3ax5W8sZFT1DFkSBEdq/G22hNCAJsAWa+9tPPwnt9d2CCXiEDcpl\n"
        "mg6hFastsoKbxfPC0gXLeqeK+xCNWa4EzktvlQC7AoGAFxQjpJ9c7NmGqVQ8TQCc\n"
        "5OLNSdsXQfcL1G/rvlqPEZAyRl4giSc/s2xOcsEmTNGyb8iYyw/4ENL85kA8OLnd\n"
        "KlRjm0wGCya+MYEneGr4JAYbtabxKaM9bXRq+lZe2r0wXGRqcXmGjtLafcUVktvD\n"
        "NtsMHJqqYvAKER29VxsdmEE=\n"
        "-----END PRIVATE KEY-----\n";

void writeTestCerts()
{
    ErrorContainer error;
    Kitsunemimi::writeFile("/tmp/cert.pem", testCert, error, true);
    Kitsunemimi::writeFile("/tmp/key.pem", testKey, error, true);
}

}

#endif // CERT_INIT_H
