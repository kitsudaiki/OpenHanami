# Installation

## On Kubernetes

For the installation on a kubernetes `helm` is used. 

!!! info

    The installation process is also very only basic at the moment. Usage of statefulsets, cert-manager and node-labels coming with version `0.2.0`. Other things later.

### Requirements

1. **Kubernetes**

    No specific version a the moment known. There are no special features used at the moment, so any version, which is not EOL should work.

    !!! example

        For fast, easy and minimal installation a `k3s` as single-node installation can be used. Installation with for example:

        ```
        curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION=v1.21.8+k3s1 sh -

        export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
        ```

2. **Helm**

    [official Installation-Guide](https://helm.sh/docs/intro/install/)

3. If measuring of the cpu power consumption should be available, then the following requirements must be fulfilled on the hosts of the kubernetes-deployment:

    - Required specific CPU-architecture:
        - **Intel**: 
            - Sandy-Bridge or newer
        - **AMD** : 
            - (**Actually broken on AMD-cpus!!!**)
            - Zen-Architecture or newer
            - for CPUs of AMD Zen/Zen2 Linux-Kernel of version `5.8` or newer must be used, for Zen3 Linux-Kernel of version `5.11` or newer

    - the `msr`-kernel module has to be loaded with `modeprobe msr`.

### Installation

```
git clone https://github.com/kitsudaiki/Hanami-AI.git

cd Hanami-AI/deploy/k8s

helm install ./hanami-ai/ --generate-name \
    --set user.id=USER_ID  \
    --set user.name=USER_NAME  \
    --set user.pw=PASSWORD  \
    --set token.pw=TOKEN_KEY
```

The `--set`-flag defining the login-information for the initial admin-user of the instance:

- `USER_ID`
    - Identifier for the new user. It is used for login and internal references to the user.
    - String, which MUST match the regex `[a-zA-Z][a-zA-Z_0-9@]*` with between `4` and `256` characters length

- `USER_NAME`
    - Better readable name for the user, which doesn't have to be unique in the system.
    - String, which MUST match the regex `[a-zA-Z][a-zA-Z_0-9 ]*` with between `4` and `256` characters length

- `PASSWORD`
    - Password for the initial user
    - String, with between `8` and `4096` characters length

- `TOKEN_KEY`
    - Key for the JWT-Tokens provided by Misaki
    - String

After a successful installation the `USER_ID` and `PASSWORD` have to be used for login to the system.

### Using

- check if all pods are running

    ```
    kubectl get pods | grep hanami

    svclb-hanami-service-hg7ht   1/1     Running   0          20m
    hanami-ai-fb996969f-5gd68    5/5     Running   2          20m
    ```

- get EXTERNAL-IP-address

    ```
    kubectl get service
    
    NAME             TYPE           CLUSTER-IP    EXTERNAL-IP       PORT(S)          AGE
    kubernetes       ClusterIP      10.43.0.1     <none>            443/TCP          44d
    hanami-service   LoadBalancer   10.43.23.42   192.168.178.110   1337:31128/TCP   28m
    ```

- use the address in your browser: `https://EXTERNAL-IP:1337`

    - `https` MUST be used at the moment, because forwarding from http to https is missing at the moment
    - port `1337` is hard configured at the moment

- login with `USER_ID` and `PASSWORD`

