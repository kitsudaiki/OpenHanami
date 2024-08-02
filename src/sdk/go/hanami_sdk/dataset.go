/**
 * @author      Tobias Anker <tobias.anker@kitsunemimi.moe>
 *
 * @copyright   Apache License Version 2.0
 *
 *      Copyright 2022 Tobias Anker
 *
 *      Licensed under the Apache License, Version 2.0 (the "License");
 *      you may not use this file except in compliance with the License.
 *      You may obtain a copy of the License at
 *
 *          http://www.apache.org/licenses/LICENSE-2.0
 *
 *      Unless required by applicable law or agreed to in writing, software
 *      distributed under the License is distributed on an "AS IS" BASIS,
 *      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *      See the License for the specific language governing permissions and
 *      limitations under the License.
 */

package hanami_sdk

import (
	"bufio"
	"encoding/json"
	"io"
	"net/url"
	"os"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/gorilla/websocket"
)

const chunkSize = 128 * 1024 // 128 KiB

func GetDataset(address, token, datasetUuid string, skipTlsVerification bool) (map[string]interface{}, error) {
	path := "v1.0alpha/dataset"
	vars := map[string]string{"uuid": datasetUuid}
	return SendGet(address, token, path, vars, skipTlsVerification)
}

func ListDataset(address, token string, skipTlsVerification bool) (map[string]interface{}, error) {
	path := "v1.0alpha/dataset/all"
	vars := map[string]string{}
	return SendGet(address, token, path, vars, skipTlsVerification)
}

func DeleteDataset(address, token, datasetUuid string, skipTlsVerification bool) (map[string]interface{}, error) {
	path := "v1.0alpha/dataset"
	vars := map[string]string{"uuid": datasetUuid}
	return SendDelete(address, token, path, vars, skipTlsVerification)
}

func CheckDataset(address, token, datasetUuid, resultDatasetUuid string, skipTlsVerification bool) (map[string]interface{}, error) {
	path := "v1.0alpha/dataset/check"
	vars := map[string]string{
		"dataset_uuid": datasetUuid,
		"result_uuid":  resultDatasetUuid,
	}
	return SendGet(address, token, path, vars, skipTlsVerification)
}

func waitUntilUploadComplete(token, address, uuid string, skipTlsVerification bool) error {
	for {
		path := "v1.0alpha/dataset/progress"
		vars := map[string]string{"uuid": uuid}
		content, err := SendGet(address, token, path, vars, skipTlsVerification)
		if err != nil {
			return err
		}

		if content["complete"].(bool) {
			return nil
		}

		time.Sleep(time.Second)
	}
}

func parseJson2(input string) map[string]interface{} {
	// parse json and fill into map
	outputMap := map[string]interface{}{}
	err := json.Unmarshal([]byte(input), &outputMap)
	if err != nil {
		panic(err)
	}

	return outputMap
}

func sendFile(token, address, fileUuid string, file *os.File) error {
	parsedURL, err := url.Parse(address)
	if err != nil {
		return err
	}

	host := parsedURL.Host
	reader := bufio.NewReader(file)

	// Create a connection to the server
	conn, _, err := websocket.DefaultDialer.Dial("ws://"+host, nil)
	if err != nil {
		return err
	}
	defer conn.Close()

	// Send the serialized message to the server
	initBody := map[string]interface{}{
		"token":  token,
		"target": "file_upload",
		"uuid":   fileUuid,
	}
	jsonData, err := json.Marshal(initBody)
	if err != nil {
		return err
	}

	err = conn.WriteMessage(websocket.TextMessage, []byte(string(jsonData)))
	if err != nil {
		return err
	}

	// Read message from WebSocket
	_, p, err := conn.ReadMessage()
	if err != nil {
		return err
	}
	response := parseJson2(string(p))
	if !response["success"].(bool) {
		return &RequestError{
			StatusCode: 403,
			Err:        "Initialize WebSocket failed",
		}
	}

	// Create a Protocol Buffers message
	message := &FileUpload_Message{}

	// Read and send chunks of data until the end of the file
	var counter uint64 = 0
	for {
		// Read a chunk of data from the file
		chunk := make([]byte, chunkSize)
		n, err := reader.Read(chunk)
		if err == io.EOF {
			break
		} else if err != nil {
			return err
		}

		// Set the chunk data in the Protocol Buffers message
		message.Position = counter * uint64(chunkSize)
		message.Data = chunk[:n]

		// Serialize the message
		data, err := proto.Marshal(message)
		if err != nil {
			return err
		}

		// Send the serialized message to the server
		err = conn.WriteMessage(websocket.BinaryMessage, []byte(data))
		if err != nil {
			return err
		}

		conn.ReadMessage()

		counter++
	}

	return nil
}

func UploadMnistFiles(address, token, name, inputFilePath, labelFilePath string, skipTlsVerification bool) (string, error) {
	// Open the binary input-file
	inputFile, err := os.Open(inputFilePath)
	if err != nil {
		return "", err
	}
	defer inputFile.Close()

	inputFileInfo, err := inputFile.Stat()
	if err != nil {
		return "", err
	}

	// Open the binary label-file
	labelFile, err := os.Open(labelFilePath)
	if err != nil {
		return "", err
	}
	defer labelFile.Close()

	labelFileInfo, err := labelFile.Stat()
	if err != nil {
		return "", err
	}

	path := "v1.0alpha/dataset/upload/mnist"
	jsonBody := map[string]interface{}{
		"name":            name,
		"input_data_size": inputFileInfo.Size(),
		"label_data_size": labelFileInfo.Size(),
	}
	datasetInfo, err := SendPost(address, token, path, jsonBody, skipTlsVerification)
	if err != nil {
		return "", err
	}

	uuid := datasetInfo["uuid"].(string)
	inputFileUuid := datasetInfo["uuid_input_file"].(string)
	labelFileUuid := datasetInfo["uuid_label_file"].(string)

	// send data
	err = sendFile(token, address, inputFileUuid, inputFile)
	if err != nil {
		return "", err
	}

	err = sendFile(token, address, labelFileUuid, labelFile)
	if err != nil {
		return "", err
	}

	err = waitUntilUploadComplete(token, address, uuid, skipTlsVerification)
	if err != nil {
		return "", err
	}

	path = "v1.0alpha/dataset/upload/mnist"
	jsonBody = map[string]interface{}{
		"uuid":            uuid,
		"uuid_input_file": inputFileUuid,
		"uuid_label_file": labelFileUuid,
	}
	_, err = SendPut(address, token, path, jsonBody, skipTlsVerification)
	if err != nil {
		return "", err
	}

	return uuid, nil
}

func UploadCsvFiles(address, token, name, inputFilePath string, skipTlsVerification bool) (string, error) {
	// Open the binary input-file
	inputFile, err := os.Open(inputFilePath)
	if err != nil {
		return "", err
	}
	defer inputFile.Close()

	inputFileInfo, err := inputFile.Stat()
	if err != nil {
		return "", err
	}

	path := "v1.0alpha/dataset/upload/csv"
	jsonBody := map[string]interface{}{
		"name":            name,
		"input_data_size": inputFileInfo.Size(),
	}
	datasetInfo, err := SendPost(address, token, path, jsonBody, skipTlsVerification)
	if err != nil {
		return "", err
	}

	uuid := datasetInfo["uuid"].(string)
	inputFileUuid := datasetInfo["uuid_input_file"].(string)

	// send data
	err = sendFile(token, address, inputFileUuid, inputFile)
	if err != nil {
		return "", err
	}

	err = waitUntilUploadComplete(token, address, uuid, skipTlsVerification)
	if err != nil {
		return "", err
	}

	path = "v1.0alpha/dataset/upload/csv"
	jsonBody = map[string]interface{}{
		"uuid":            uuid,
		"uuid_input_file": inputFileUuid,
	}
	_, err = SendPut(address, token, path, jsonBody, skipTlsVerification)
	if err != nil {
		return "", err
	}

	return uuid, nil
}
