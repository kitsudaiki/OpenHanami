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

from .hanami_messages import proto3_pb2


def send_train_input(ws, brick_name, values, is_last, verify_connection: bool = True):

    cluster_io_msg = proto3_pb2.ClusterIO_Message()
    cluster_io_msg.brickName = brick_name
    cluster_io_msg.isLast = is_last
    cluster_io_msg.processType = proto3_pb2.ClusterProcessType.TRAIN_TYPE
    cluster_io_msg.numberOfValues = len(values)
    cluster_io_msg.values.extend(values)
    ws.send(cluster_io_msg.SerializeToString())
    ws.recv()


def send_request_input(ws, brick_name, values, is_last, verify_connection: bool = True):

    cluster_io_msg = proto3_pb2.ClusterIO_Message()
    cluster_io_msg.brickName = brick_name
    cluster_io_msg.isLast = is_last
    cluster_io_msg.processType = proto3_pb2.ClusterProcessType.REQUEST_TYPE
    cluster_io_msg.numberOfValues = len(values)
    cluster_io_msg.values.extend(values)
    ws.send(cluster_io_msg.SerializeToString())

    response = ws.recv()

    if is_last:
        response_message = proto3_pb2.ClusterIO_Message()
        response_message.ParseFromString(response)

        return response_message.values
    else:
        return None
