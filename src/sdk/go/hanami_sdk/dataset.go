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
    "fmt"
    "io"
    "os"
    "bufio"
    "time"
    "net/url"
    "encoding/json"

    "github.com/gorilla/websocket"
    "github.com/golang/protobuf/proto"
)

const chunkSize = 128 * 1024 // 128 KiB


func parseJson2(input string) map[string]interface{} {
    // parse json and fill into map
    outputMap := map[string]interface{}{}
    err := json.Unmarshal([]byte(input), &outputMap)
    if err != nil {
        panic(err)
    }

    return outputMap
}

func GetDataset(address string, token string, datasetUuid string) (bool, string) {
    path := "control/v1.0alpha/dataset"
    vars := fmt.Sprintf("uuid=%s", datasetUuid)
    return SendGet(address, token, path, vars)
}

func ListDataset(address string, token string) (bool, string) {
    path := fmt.Sprintf("control/v1.0alpha/dataset/all")
    vars := ""
    return SendGet(address, token, path, vars)
}

func DeleteDataset(address string, token string, datasetUuid string) (bool, string) {
    path := "control/v1.0alpha/dataset"
    vars := fmt.Sprintf("uuid=%s", datasetUuid)
    return SendDelete(address, token, path, vars)
}
 
func waitUntilUploadComplete(token string, address string, uuid string) bool {
    for true {
        path := fmt.Sprintf("control/v1.0alpha/dataset/progress")
        vars := fmt.Sprintf("uuid=%s", uuid)
        success, content := SendGet(address, token, path, vars)
        if success == false {
            fmt.Println("fail: ", content)
            return false
        }

        outputMap := parseJson2(content)
        if outputMap["complete"].(bool) {
            return true
        }

        time.Sleep(time.Second)
    }

    return true
}


func sendFile(token string, address string, datasetUuid string, fileUuid string, file *os.File) bool {
    // Parse the URL
    parsedURL, err := url.Parse(address)
    if err != nil {
        fmt.Println("Error parsing URL:", err)
        return false
    }

    // Extract the host
    host := parsedURL.Host

    // Create a buffered reader for efficient reading
    reader := bufio.NewReader(file)

    // Create a connection to the server
    conn, _, err := websocket.DefaultDialer.Dial("ws://" + host, nil)
    if err != nil {
        fmt.Println("Error connecting to server:", err)
        return false
    }
    defer conn.Close()


    // Send the serialized message to the server
    initBody := fmt.Sprintf("{\"token\":\"%s\", \"target\":\"file_upload\", \"uuid\":\"%s\"}", token, fileUuid)
    err = conn.WriteMessage(websocket.TextMessage, []byte(initBody))
    if err != nil {
        fmt.Println("Error sending data to server:", err)
        return false
    }

    // Read message from WebSocket
    _, p, err := conn.ReadMessage()
    if err != nil {
        return false
    }
    response := parseJson2(string(p))
    if response["success"].(bool) == false {
        return false
    }

    // Create a Protocol Buffers message
    message := &FileUpload_Message{}

    // Read and send chunks of data until the end of the file
    var counter uint64
    counter = 0
    for {
        // Read a chunk of data from the file
        chunk := make([]byte, chunkSize)
        n, err := reader.Read(chunk)
        if err == io.EOF {
            break;
        } else if err != nil {
            fmt.Println("Error reading from file:", err)
            return false
        }

        // Set the chunk data in the Protocol Buffers message
        message.Position = counter * uint64(chunkSize)
        message.Data = chunk[:n]

        // Serialize the message
        data, err := proto.Marshal(message)
        if err != nil {
            fmt.Println("Error marshaling protobuf message:", err)
            return false
        }

        // Send the serialized message to the server
        err = conn.WriteMessage(websocket.BinaryMessage, []byte(data))
        if err != nil {
            fmt.Println("Error sending data to server:", err)
            return false
        }

        conn.ReadMessage()

        counter++;
    }

    return true
}

func UploadMnistFiles(address string, token string, name string, inputFilePath string, labelFilePath string) (bool, string) {
    // Open the binary input-file
    inputFile, err := os.Open(inputFilePath)
    if err != nil {
        fmt.Printf("Error opening file: %v\n", err)
        return false, ""
    }
    defer inputFile.Close()

    inputFileInfo, err := inputFile.Stat()
    if err != nil {
        fmt.Println("Error getting file info:", err)
        return false, ""
    }

    // Open the binary label-file
    labelFile, err := os.Open(labelFilePath)
    if err != nil {
        fmt.Printf("Error opening file: %v\n", err)
        return false, ""
    }
    defer labelFile.Close()

    labelFileInfo, err := labelFile.Stat()
    if err != nil {
        fmt.Println("Error getting file info:", err)
        return false, ""
    }

    jsonBody := fmt.Sprintf("{\"name\":\"%s\", \"input_data_size\":%d, \"label_data_size\":%d}", name, inputFileInfo.Size(), labelFileInfo.Size())
    path := "control/v1.0alpha/mnist/dataset"
    vars := ""
    success, content := SendPost(address, token, path, vars, jsonBody)
    if success == false {
        fmt.Println("Failed to init file-upload:", content)
        return false, ""
    }

    datasetInfo := parseJson2(content)
    uuid := datasetInfo["uuid"].(string)
    inputFileUuid := datasetInfo["uuid_input_file"].(string)
    labelFileUuid := datasetInfo["uuid_label_file"].(string)

    // send data
    if sendFile(token, address, uuid, inputFileUuid, inputFile) == false {
        return false, ""
    }

    if sendFile(token, address, uuid, labelFileUuid, labelFile) == false {
        return false, ""
    }

    if waitUntilUploadComplete(token, address, uuid) == false {
        return false, ""
    }

    jsonBody = fmt.Sprintf("{\"uuid\":\"%s\", \"uuid_input_file\":\"%s\", \"uuid_label_file\":\"%s\"}", uuid, inputFileUuid, labelFileUuid)
    path = "control/v1.0alpha/mnist/dataset"
    vars = ""
    success, content = SendPut(address, token, path, vars, jsonBody)
    if success == false {
        fmt.Println("Failed to finalize file-upload:", content)
        return false, ""
    }

    return true, uuid
}

func UploadCsvFiles(address string, token string, name string, inputFilePath string) (bool, string) {
    // Open the binary input-file
    inputFile, err := os.Open(inputFilePath)
    if err != nil {
        fmt.Printf("Error opening file: %v\n", err)
        return false, ""
    }
    defer inputFile.Close()

    inputFileInfo, err := inputFile.Stat()
    if err != nil {
        fmt.Println("Error getting file info:", err)
        return false, ""
    }

    jsonBody := fmt.Sprintf("{\"name\":\"%s\", \"input_data_size\":%d}", name, inputFileInfo.Size())
    path := "control/v1.0alpha/csv/dataset"
    vars := ""
    success, content := SendPost(address, token, path, vars, jsonBody)
    if success == false {
        fmt.Println("Failed to init file-upload:", content)
        return false, ""
    }

    datasetInfo := parseJson2(content)
    uuid := datasetInfo["uuid"].(string)
    inputFileUuid := datasetInfo["uuid_input_file"].(string)

    // send data
    if sendFile(token, address, uuid, inputFileUuid, inputFile) == false {
        return false, ""
    }

    if waitUntilUploadComplete(token, address, uuid) == false {
        return false, ""
    }

    jsonBody = fmt.Sprintf("{\"uuid\":\"%s\", \"uuid_input_file\":\"%s\"}", uuid, inputFileUuid)
    path = "control/v1.0alpha/csv/dataset"
    vars = ""
    success, content = SendPut(address, token, path, vars, jsonBody)
    if success == false {
        fmt.Println("Failed to finalize file-upload:", content)
        return false, ""
    }

    return true, uuid
}

