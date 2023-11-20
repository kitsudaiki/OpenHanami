# Copyright 2022 Tobias Anker
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from websockets.sync.client import connect
from hanami_sdk.hanami_messages import proto3_pb2
import asyncio


def send_train_data(ws, input_values, should_values):

    cluster_io_msg = proto3_pb2.ClusterIO_Message()
    cluster_io_msg.segmentName = "input"
    cluster_io_msg.isLast = False
    cluster_io_msg.processType = proto3_pb2.ClusterProcessType.TRAIN_TYPE
    cluster_io_msg.dataType = proto3_pb2.ClusterDataType.INPUT_TYPE
    cluster_io_msg.numberOfValues = len(input_values)
    cluster_io_msg.values.extend(input_values)
    serialized_msg = cluster_io_msg.SerializeToString()

    ws.send(serialized_msg)
    message = ws.recv()

    cluster_io_msg = proto3_pb2.ClusterIO_Message()
    cluster_io_msg.segmentName = "output"
    cluster_io_msg.isLast = True
    cluster_io_msg.processType = proto3_pb2.ClusterProcessType.TRAIN_TYPE
    cluster_io_msg.dataType = proto3_pb2.ClusterDataType.SHOULD_TYPE
    cluster_io_msg.numberOfValues = len(should_values)
    cluster_io_msg.values.extend(should_values)
    serialized_msg = cluster_io_msg.SerializeToString()

    ws.send(serialized_msg)
    message = ws.recv()


def send_request_data(ws, input_values):

    cluster_io_msg = proto3_pb2.ClusterIO_Message()
    cluster_io_msg.segmentName = "input"
    cluster_io_msg.isLast = True
    cluster_io_msg.processType = proto3_pb2.ClusterProcessType.REQUEST_TYPE
    cluster_io_msg.dataType = proto3_pb2.ClusterDataType.INPUT_TYPE
    cluster_io_msg.numberOfValues = len(input_values)
    cluster_io_msg.values.extend(input_values)
    serialized_msg = cluster_io_msg.SerializeToString()

    ws.send(serialized_msg)

    response = ws.recv()
    response_message = proto3_pb2.ClusterIO_Message()
    response_message.ParseFromString(response)

    return response_message.values
