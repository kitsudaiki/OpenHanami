#!/bin/bash

export HANAMI_ADDRESS=http://127.0.0.1:11418
export HANAMI_USER=asdf
export HANAMI_PW=asdfasdf

export train_inputs=/tmp/train-images-idx3-ubyte
export train_labels=/tmp/train-labels-idx1-ubyte
export request_inputs=/tmp/t10k-images-idx3-ubyte
export request_labels=/tmp/t10k-labels-idx1-ubyte


# build protobuffer for go sdk
# pushd ../../src/sdk/go/hanami_sdk 
# protoc --go_out=. --proto_path ../../../libraries/hanami_messages/protobuffers hanami_messages.proto3
# popd

# build cli-binarygolangci-lint
# pushd ../../src/cli/hanamictl
# go build .
# popd
# cp ../../src/cli/hanamictl/hanamictl .

# cleanup before running tests
./hanamictl project delete cli_test_project
./hanamictl user delete cli_test_user

########################
echo ""
echo "project tests"
./hanamictl project create -n "cli test project" cli_test_project
./hanamictl project get cli_test_project
./hanamictl project list
./hanamictl project delete cli_test_project

########################
echo ""
echo "user tests"
./hanamictl user create -n "cli test user" -p "asdfasdfasdf" cli_test_user
./hanamictl user get cli_test_user
./hanamictl user list
./hanamictl user delete cli_test_user

########################
echo ""
echo "dataset tests"
dataset_uuid=$(./hanamictl dataset create mnist -j -i $train_inputs -l $train_labels cli_test_dataset | jq -r '.uuid')
./hanamictl dataset get $dataset_uuid
./hanamictl dataset list
./hanamictl dataset delete $dataset_uuid

########################
echo ""
echo "cluster tests"
cluster_uuid=$(./hanamictl cluster create -j -t ./cluster_template cli_test_cluster | jq -r '.uuid')
./hanamictl cluster get $cluster_uuid
./hanamictl cluster list
./hanamictl cluster delete $cluster_uuid


########################
echo ""
echo "workfloat tests"
./hanamictl host list 

train_dataset_uuid=$(./hanamictl dataset create mnist -j -i $train_inputs -l $train_labels cli_test_dataset_train | jq -r '.uuid')
echo "Train-Dataset-UUID: $train_dataset_uuid"

request_dataset_uuid=$(./hanamictl dataset create mnist -j -i $request_inputs -l $request_labels cli_test_dataset_req | jq -r '.uuid')
echo "Request-Dataset-UUID: $request_dataset_uuid"

cluster_uuid=$(./hanamictl cluster create -j -t ./cluster_template cli_test_cluster | jq -r '.uuid')
echo "Cluster-UUID: $cluster_uuid"


# train test
task_uuid=$(./hanamictl task create train -j -i picture:$train_dataset_uuid -o label:$train_dataset_uuid -c $cluster_uuid cli_train_test_task | jq -r '.uuid')
echo "Train-Task-UUID: $task_uuid"

while true; do
    ./hanamictl task get -c $cluster_uuid $task_uuid
    state=$(./hanamictl task get -j -c $cluster_uuid $task_uuid | jq -r '.state')
    if [[ "$state" == *"finished"* ]]; then
        echo "Process finished. Exiting loop."
        break
    fi
    sleep 1
done
./hanamictl task get -c $cluster_uuid $task_uuid


# save and restore test
./hanamictl cluster save -n cli_test_checkpoint $cluster_uuid
checkpoint_uuid=$(./hanamictl cluster save -j -n cli_test_checkpoint $cluster_uuid  | jq -r '.uuid')
sleep 1
echo "Checkpoint-UUID: $checkpoint_uuid"
./hanamictl checkpoint list

./hanamictl cluster delete $cluster_uuid
cluster_uuid=$(./hanamictl cluster create -j -t ./cluster_template cli_test_cluster | jq -r '.uuid')
echo "new Cluster-UUID: $cluster_uuid"
./hanamictl cluster restore -c $checkpoint_uuid $cluster_uuid

# request test
req_task_uuid=$(./hanamictl task create request -j -i picture:$request_dataset_uuid -r label:cli_test_output -c $cluster_uuid cli_request_test_task | jq -r '.uuid')
echo "Request-Task-UUID: $req_task_uuid"

./hanamictl task list -c $cluster_uuid 
echo "$taskUuid"
./hanamictl task delete -c $cluster_uuid $task_uuid

while true; do
    ./hanamictl task get -c $cluster_uuid $req_task_uuid
    state=$(./hanamictl task get -j -c $cluster_uuid $req_task_uuid | jq -r '.state')
    if [[ "$state" == *"finished"* ]]; then
        echo "Process finished. Exiting loop."
        break
    fi
    sleep 1
done
./hanamictl task get -c $cluster_uuid $req_task_uuid

./hanamictl dataset list


result_uuid=$(./hanamictl dataset list -j | jq -r '.body[] | select(.[-1] == "cli_test_output") | .[1]')
echo "Result-Dataset-UUID: $result_uuid"

./hanamictl dataset get $result_uuid

./hanamictl dataset check -r $request_dataset_uuid $result_uuid


# clear all test-resources
./hanamictl checkpoint delete $checkpoint_uuid
./hanamictl cluster delete $cluster_uuid
./hanamictl dataset delete $train_dataset_uuid
./hanamictl dataset delete $request_dataset_uuid
./hanamictl dataset delete $result_uuid
