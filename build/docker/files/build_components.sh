REGISTRY=kitsudaiki
GIT_TAG="style/create-monorepo-by-replacing-submodules"
DOCKER_TAG="test"

# base
cd /dockerbuilder
docker build -t $REGISTRY/hanami_ai_base:$DOCKER_TAG -f Dockerfile_base .

git clone https://github.com/kitsudaiki/Hanami-AI.git
cd /dockerbuilder/Hanami-AI
git checkout $GIT_TAG
git submodule init
git submodule update --recursive
./build.sh
cd /dockerbuilder

docker build -t $REGISTRY/kyouko_mind:$DOCKER_TAG -f Dockerfile_kyouko .
docker build -t $REGISTRY/misaki_guard:$DOCKER_TAG -f Dockerfile_misaki .
docker build -t $REGISTRY/azuki_heart:$DOCKER_TAG -f Dockerfile_azuki .
docker build -t $REGISTRY/shiori_archive:$DOCKER_TAG -f Dockerfile_shiori .
docker build -t $REGISTRY/torii_gateway:$DOCKER_TAG -f Dockerfile_torii .
docker build -t $REGISTRY/hanami_ai_dashboard:$DOCKER_TAG -f Dockerfile_dashboard .

