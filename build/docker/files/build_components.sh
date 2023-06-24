REGISTRY="kitsudaiki"
GIT_TAG="develop"
DOCKER_TAG="develop"

# base
cd /dockerbuilder
docker build -t kitsudaiki/hanami_ai_base:develop -f Dockerfile_base .

git clone https://github.com/kitsudaiki/Hanami-AI.git
cd /dockerbuilder/Hanami-AI
git checkout $GIT_TAG
git submodule init
git submodule update --recursive
./build.sh
cd /dockerbuilder

docker build -t $REGISTRY/hanami:$DOCKER_TAG -f Dockerfile_hanami .
docker build -t $REGISTRY/hanami_ai_dashboard:$DOCKER_TAG -f Dockerfile_dashboard .

