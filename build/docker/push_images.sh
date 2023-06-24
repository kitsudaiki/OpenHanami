REGISTRY="kitsudaiki"
TAG=develop

# HINT (kitsudaiki): base-image is only pushed for the ci-pipeline,
#                    so it always has the develop-tag. So this is not a bug.
docker push $REGISTRY/hanami_ai_base:develop

docker push $REGISTRY/hanami:$TAG
docker push $REGISTRY/hanami_ai_dashboard:$TAG
