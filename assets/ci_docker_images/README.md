Docker-image for the build-tests within the ci-pipeline 

Build:

```
docker build -t kitsudaiki/ci-build:0.1.0 .
docker push kitsudaiki/ci-build:0.1.0
```