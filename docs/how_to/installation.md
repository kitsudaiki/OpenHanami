# Installation

## On Kubernetes

For the installation on a kubernetes `helm` is used. 

!!! info

    The installation process is also very basic at the moment.

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

3. **Cert-Manager**

    Installation:

    ```
    helm repo add jetstack https://charts.jetstack.io
    helm repo update
    kubectl create namespace cert-manager
    helm install cert-manager jetstack/cert-manager --namespace cert-manager --set installCRDs=true
    ```

4. **Node label**

    To all avaialbe nodes, where it is allowed to be deployed, the label `hanami-node` must be assigned

    ```
    kubectl label nodes NODE_NAME hanami-node=true
    ```

    !!! info

        At the moment Hanami is only a single-node application. This will change in the near future, but at the moment it doesn't make sense to label more than one node.

<!-- 3. If measuring of the cpu power consumption should be available, then the following requirements must be fulfilled on the hosts of the kubernetes-deployment:

    - Required specific CPU-architecture:
        - **Intel**: 
            - Sandy-Bridge or newer
        - **AMD** : 
            - Zen-Architecture or newer
            - for CPUs of AMD Zen/Zen2 Linux-Kernel of version `5.8` or newer must be used, for Zen3 Linux-Kernel of version `5.11` or newer

    - the `msr`-kernel module has to be loaded with `modeprobe msr`. -->

### Installation

```
git clone https://github.com/kitsudaiki/Hanami.git

cd Hanami/deploy/k8s

helm install \
    --set user.id=USER_ID  \
    --set user.name=USER_NAME  \
    --set user.pw=PASSWORD  \
    --set token.pw=TOKEN_KEY  \
    --set api.domain=DOMAIN_NAME  \
    hanami \
    ./hanami/
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
    - Key for the JWT-Tokens
    - String

- `DOMAIN_NAME`
    - Domain for https-access. Per default it is `local-hanami`
    - String

After a successful installation the `USER_ID` and `PASSWORD` have to be used for login to the system.

### Using

- check if all pods are running

    !!! example
    
        ```
        kubectl get pods

        NAME                      READY   STATUS    RESTARTS   AGE
        hanami-56fc87c8f5-6k77r   1/1     Running   0          14s
        ```

- get IP-address

    !!! example
    
        ```
        kubectl get ingress

        NAME                      CLASS     HOSTS          ADDRESS          PORTS     AGE
        hanami-ingress-redirect   traefik   local-hanami   192.168.178.87   80        8s
        hanami-ingress            traefik   local-hanami   192.168.178.87   80, 443   8s
        ```

- add domain with ip to `/etc/hosts`

    !!! example

        ```
        192.168.178.87  local-hanami
        ```

- use the address in your browser: 

    `https://DOMAIN_NAME`

    !!! example
    
        ```
        https://local-hanami/
        ```

- login with `USER_ID` and `PASSWORD`

