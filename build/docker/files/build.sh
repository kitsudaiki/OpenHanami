REGISTRY=kitsudaiki
TAG=develop


# base
cd /dockerbuilder
docker build -t $REGISTRY/hanami_ai_base:$TAG -f Dockerfile_base .


# kyouko
git clone https://github.com/kitsudaiki/KyoukoMind.git /dockerbuilder/KyoukoMind
cd /dockerbuilder/KyoukoMind
./build.sh
cd /dockerbuilder
docker build -t $REGISTRY/kyouko_mind:$TAG -f Dockerfile_kyouko .

# misaki
git clone https://github.com/kitsudaiki/MisakiGuard.git /dockerbuilder/MisakiGuard
cd /dockerbuilder/MisakiGuard
./build.sh
cd /dockerbuilder
docker build -t $REGISTRY/misaki_guard:$TAG -f Dockerfile_misaki .

# azuki
git clone https://github.com/kitsudaiki/AzukiHeart.git /dockerbuilder/AzukiHeart
cd /dockerbuilder/AzukiHeart
./build.sh
cd /dockerbuilder
docker build -t $REGISTRY/azuki_heart:$TAG -f Dockerfile_azuki .

# shiori
git clone https://github.com/kitsudaiki/ShioriArchive.git /dockerbuilder/ShioriArchive
cd /dockerbuilder/ShioriArchive
./build.sh
cd /dockerbuilder
docker build -t $REGISTRY/shiori_archive:$TAG -f Dockerfile_shiori .

# torii
git clone https://github.com/kitsudaiki/ToriiGateway.git /dockerbuilder/ToriiGateway
cd /dockerbuilder/ToriiGateway
./build.sh
cd /dockerbuilder
docker build -t $REGISTRY/torii_gateway:$TAG -f Dockerfile_torii .

# dashboard
cd /dockerbuilder
git clone https://github.com/kitsudaiki/Hanami-AI-Dashboard.git /dockerbuilder/Hanami-AI-Dashboard
cd /dockerbuilder/Hanami-AI-Dashboard/src
git clone https://github.com/kitsudaiki/libHanamiAiSdk.git
git clone https://github.com/kitsudaiki/libKitsunemimiHanamiMessages.git
git clone https://github.com/kitsudaiki/Hanami-AI-Dashboard-Dependencies.git
cd /dockerbuilder
docker build -t $REGISTRY/hanami_ai_dashboard:$TAG -f Dockerfile_dashboard .
