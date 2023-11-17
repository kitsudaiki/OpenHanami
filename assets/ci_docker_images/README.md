Docker-image for the build-tests within the ci-pipeline 

Build:

```
docker build -f Dockerfile_check -t kitsudaiki/ci-check:0.1.0 .
docker push kitsudaiki/ci-check:0.1.0
```
