# Overview

!!! info "**INPORTANT**"

    In order to not wait any longer to bring the project finally open-source, the documentation is still very basic and many internal details are not documented here yet. It will be updated over time.

![Overview](../img/overview.png)
<!-- I had do use a image instead of the drawio, because for an unknown reason the blossom was not printed with the drawio-exporter -->


??? question "Why were the components named like this?"

    The names are all a combination of a female japanese name, which sounds good for me and are not too long or too short, together with an english word, which is related to the content of the component. The only exception is the [Torii](/internal/3_torii/#torii). For the japanese names it was hard to decide. I tried to use names with different first character for easier separation of the component and reducing the list of possible names. 

!!! info

    Because it sounds better, when a component is referenced in the documents, only the prefix of the name is used, so `Kyouko` instead of `KyoukoMind`. The long version for the repository names has the same reason, like for the libraries, only to make it more clear, which component exist for which task. Beside this, the components are referenced in the documentation also by `she` or `her` instead of `it`, because this sounds more natural in regard of the female names.

- **Kyouko**
    - Content: Core-component, which holds the artificial neuronal networks.
    - Documentation: [Kyouko-internal](/Inner_Workings/3_kyouko/).
    - Repository: https://github.com/kitsudaiki/KyoukoMind.git

- **Misaki**
    - Content: Authentication-service and management of user
    - Documentation: [Misaki-internal](/Inner_Workings/6_misaki/).
    - Repository: https://github.com/kitsudaiki/MisakiGuard.git

- **Shiori**
    - Content: Storage-component, which holds snapshots, logs and so on
    - Documentation: [Shiori-internal](/Inner_Workings/5_shiori/).
    - Repository: https://github.com/kitsudaiki/ShioriArchive.git

- **Azuki**
    - Content: Monitoring and energy-optimization
    - Documentation: [Azuki-internal](/Inner_Workings/4_azuki/).
    - Repository: https://github.com/kitsudaiki/AzukiHeart.git

- **Torii**
    - Content: Proxy for all incoming connections
    - Documentation: [Torii-internal](/Inner_Workings/7_torii/).
    - Repository: https://github.com/kitsudaiki/ToriiGateway.git

- **Tsugumi**
    - Content: Test-Tool to test the API-endpoints with the help of the SDK-library
    - Repository: https://github.com/kitsudaiki/TsugumiTester.git

!!! info

    Two components (the empty bubbles) are planned, but not implemented at the moment. They are necessary for the desired concept to give the whole project the capability of Multi-Node-Setups.

!!! info

    Tests with Tsugumi are done locally at the moment, but in the future there should be also a CI-pipline, which runs the tests.
