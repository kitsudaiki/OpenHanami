REGISTRY="kitsudaiki"
GIT_TAG="develop"
DOCKER_TAG="develop"

# base
cd /dockerbuilder
docker build -t kitsudaiki/hanami_base:develop -f Dockerfile_hanami_base .

git clone https://github.com/kitsudaiki/Hanami.git
cd /dockerbuilder/Hanami
git checkout $GIT_TAG
git submodule init
git submodule update --recursive
./build.sh
cd /dockerbuilder

docker build -t $REGISTRY/hanami:$DOCKER_TAG -f Dockerfile_hanami .

