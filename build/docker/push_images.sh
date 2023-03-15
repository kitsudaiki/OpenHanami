REGISTRY="kitsudaiki"
TAG=0.2.0

# HINT (kitsudaiki): base-image is only pushed for the ci-pipeline,
#                    so it always has the develop-tag. So this is not a bug.
docker push $REGISTRY/hanami_ai_base:develop

docker push $REGISTRY/kyouko_mind:$TAG
docker push $REGISTRY/misaki_guard:$TAG
docker push $REGISTRY/azuki_heart:$TAG
docker push $REGISTRY/shiori_archive:$TAG
docker push $REGISTRY/torii_gateway:$TAG
docker push $REGISTRY/hanami_ai_dashboard:$TAG
