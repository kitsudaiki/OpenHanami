# Installation

!!! warning

    The installation process is also very basic at the moment. So for example only self-signed certificates are used at the moment. Because of this the given SDK also has a problem to connect correctly to these installation currently.

## On Kubernetes

For the installation on a kubernetes `helm` is used.

| Supported versions                                          |
| --------------------------------------------------- |
| [![kubernetes-1_26][img_kubernetes-1_26]][Workflow] |
| [![kubernetes-1_27][img_kubernetes-1_27]][Workflow] |
| [![kubernetes-1_28][img_kubernetes-1_28]][Workflow] |
| [![kubernetes-1_29][img_kubernetes-1_29]][Workflow] |
| [![kubernetes-1_30][img_kubernetes-1_30]][Workflow] |                                                   |

[img_kubernetes-1_26]:
    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/kitsudaiki/OpenHanami-badges/develop/kubernetes_version/kubernetes-1_26/shields.json&style=flat-square
[img_kubernetes-1_27]:
    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/kitsudaiki/OpenHanami-badges/develop/kubernetes_version/kubernetes-1_27/shields.json&style=flat-square
[img_kubernetes-1_28]:
    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/kitsudaiki/OpenHanami-badges/develop/kubernetes_version/kubernetes-1_28/shields.json&style=flat-square
[img_kubernetes-1_29]:
    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/kitsudaiki/OpenHanami-badges/develop/kubernetes_version/kubernetes-1_29/shields.json&style=flat-square
[img_kubernetes-1_30]:
    https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/kitsudaiki/OpenHanami-badges/develop/kubernetes_version/kubernetes-1_30/shields.json&style=flat-square
[Workflow]: https://github.com/kitsudaiki/OpenHanami/actions/workflows/build_test.yml

### Requirements

1.  **Kubernetes**

    No specific version a the moment known. There are no special features used at the moment, so any
    version, which is not EOL should work.

    !!! example

        For fast, easy and minimal installation a `k3s` as single-node installation can be used. Installation with for example:

        ```
        curl -sfL https://get.k3s.io | sh -

        export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
        ```

2.  **Helm**

    [official Installation-Guide](https://helm.sh/docs/intro/install/)

3.  **Cert-Manager**

    Installation:

    ```
    helm repo add jetstack https://charts.jetstack.io
    helm repo update
    kubectl create namespace cert-manager
    helm install cert-manager jetstack/cert-manager --namespace cert-manager --set installCRDs=true
    ```

4.  **Node label**

    To all avaialbe nodes, where it is allowed to be deployed, the label `hanami-node` must be
    assigned

    ```
    kubectl label nodes NODE_NAME hanami-node=true
    ```

    !!! info

        At the moment OpenHanami is only a single-node application. This will change in the near future, but at the moment it doesn't make sense to label more than one node.

<!-- 3. If measuring of the cpu power consumption should be available, then the following requirements must be fulfilled on the hosts of the kubernetes-deployment:

    - Required specific CPU-architecture:
        - **Intel**:
            - Sandy-Bridge or newer
        - **AMD** :
            - Zen-Architecture or newer
            - for CPUs of AMD Zen/Zen2 Linux-Kernel of version `5.8` or newer must be used, for Zen3 Linux-Kernel of version `5.11` or newer

    - the `msr`-kernel module has to be loaded with `modeprobe msr`. -->

### Installation

**From repository**

```bash
git clone https://github.com/kitsudaiki/OpenHanami.git

cd OpenHanami/deploy/k8s

helm install \
    --set docker.tag=DOCKER_IMAGE_TAG \
    --set user.id=USER_ID  \
    --set user.name=USER_NAME  \
    --set user.pw=PASSWORD  \
    --set token.pw=TOKEN_KEY  \
    --set api.domain=DOMAIN_NAME  \
    openhanami \
    ./openhanami/
```

**From pre-build**

Download the helm-chart from the [file-share](https://files.hanami-ai.com/)

```bash
helm install \
    --set docker.tag=DOCKER_IMAGE_TAG \
    --set user.id=USER_ID  \
    --set user.name=USER_NAME  \
    --set user.pw=PASSWORD  \
    --set token.pw=TOKEN_KEY  \
    --set api.domain=DOMAIN_NAME  \
    openhanami \
    openhanami-x.y.z.tgz
```

The `--set`-flag defining the login-information for the initial admin-user of the instance:

-   `USER_ID`

    -   Identifier for the new user. It is used for login and internal references to the user.
    -   String, which MUST match the regex `[a-zA-Z][a-zA-Z_0-9@]*` with between `4` and `256`
        characters length

-   `USER_NAME`

    -   Better readable name for the user, which doesn't have to be unique in the system.
    -   String, which MUST match the regex `[a-zA-Z][a-zA-Z_0-9 ]*` with between `4` and `256`
        characters length

-   `PASSWORD`

    -   Password for the initial user
    -   String, with between `8` and `4096` characters length

-   `TOKEN_KEY`

    -   Key for the JWT-Tokens
    -   String

-   `DOMAIN_NAME`

    -   Domain for https-access. Per default it is `local-hanami`
    -   String

-   `DOCKER_IMAGE_TAG`
    -   Docker-tag used from
        [docker-hub](https://hub.docker.com/repository/docker/kitsudaiki/hanami/tags)
    -   String

After a successful installation the `USER_ID` and `PASSWORD` have to be used for login to the
system.

### Using

-   check if all pods are running

    !!! example

          ```bash
          kubectl get pods

          NAME                      READY   STATUS    RESTARTS   AGE
          hanami-56fc87c8f5-6k77r   1/1     Running   0          14s
          ```

-   get IP-address

    !!! example

          ```bash
          kubectl get ingress

          NAME                      CLASS     HOSTS          ADDRESS          PORTS     AGE
          hanami-ingress-redirect   traefik   local-hanami   192.168.178.87   80        8s
          hanami-ingress            traefik   local-hanami   192.168.178.87   80, 443   8s
          ```

-   add domain with ip to `/etc/hosts`

    !!! example

          ```
          192.168.178.87  local-hanami
          ```

-   use the address in your browser:

    `https://DOMAIN_NAME`

    !!! example

          ```
          https://local-hanami/
          ```

-   login with `USER_ID` and `PASSWORD`

    !!! note

        Persistent data of hanami (checkpoints and so on) within the pod are mounted into the host directory `/etc/openhanami_data`, in case you want to save or delete them.

## With Ansible

### Requirements

-   Only Ansible itself is required to be installed

-   Other dependencies like docker will be installed by the ansible playbooks, so it is required
    that playbooks can be executed with sudo permissions

### Environment Variables

Basic parameter have to be set by environemt-variables:

-   `ADMIN_USER_ID`

    -   Identifier for the new user. It is used for login and internal references to the user.
    -   String, which MUST match the regex `[a-zA-Z][a-zA-Z_0-9@]*` with between `4` and `256`
        characters length

-   `ADMIN_USER_NAME`

    -   Better readable name for the user, which doesn't have to be unique in the system.
    -   String, which MUST match the regex `[a-zA-Z][a-zA-Z_0-9 ]*` with between `4` and `256`
        characters length

-   `ADMIN_PASSWORD`

    -   Password for the initial user
    -   String, with between `8` and `4096` characters length

-   `TOKEN_KEY`
    -   Key for the JWT-Tokens
    -   String

!!! example

    ```bash
    export ADMIN_USER_ID="admin"
    export ADMIN_USER_NAME="admin"
    export ADMIN_PASSWORD="some_password"
    export TOKEN_KEY="random_token_key"
    ```

### Run

Run in the root of the repository:

```bash
cd deploy/ansible/

ansible-playbook --connection=local -i openhanami/inventory.yml openhanami/deploy.yml
```

The resulting setup will listen on `0.0.0.0` and port `443` and `80`.

### Testing

The playbooks can be tested with `vagrant`

```bash
apt-get install vagrant virtualbox

vagrant plugin install vagrant-env
vagrant plugin install vagrant-vbguest
```

The environment variables also must be set in this case with the initial admin credentials.

Run in the root of the repository:

```bash
cd testing/ansible_deploy

vagrant up
```

It will create a virtualbox-VM with ubuntu 22.04 and automatically deploy OpenHanami with the
ansible-playbook.
