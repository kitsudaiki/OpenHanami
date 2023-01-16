# Glossar

## Core-Structure

### **Neuron**

`Neuron` are the smalles entity within the core-structure, which provides the base of the neuronal network. A `Neuron` is connected one or more `Synapses` to other nodes. 

### **Synapse**

`Synapses` are the connetions between the nodes. They are not static and can be create, destroyed and modified.

### **Brick**

A `Brick` consist of multiple nodes, which are connected by synapses ot nodes in other bricks. There are basically 3 types:

1. `Input-Bricks`

2. `Output-Bricks`

3. `Core-Bricks`

### **Segment**

A `Segment` it the atomic entity and consist of multiple `Bricks` with at least one `Input-Brick` and `Output-Brick`. Here is also the 3 type separation:

1. `Input-Segments`
    
    Connected to the input-buffer of the segment.

2. `Output-Segments`

    Connected to the output-buffer of the segment.

3. `Core-Segments`

    1. `Dynamic Segments`

        Segment of the actual core-concept with dynamic creation of connctions between artificial neurons.

### **Cluster**

`Cluster` are again a collection of multiple `Segments`

### **Data-Set**

A `Data-Set` defines a set of train-data or data, which should be used for requests against the neuronal network. This can be a table in CSV-format or an MNIST-dataset for now.

## Infrastrure

### Tasks

Tasks an asynchronous operation of the network based on a given `Data-Set`. There are two types:

1. `Learn-Task`

For `Learn-Tasks` input-data and desired output must exist in the `Data-Set` in order to update the neuronal network based on the data.

2. `Request-Task`

In `Request-Tasks` only the input-data are provided for the network in order to generate a output of the previouly trained network. The output is stored in `Shiori` as `Request-Result`.

### **Cluster-Snapshot** or **Snapshot**

`Cluster-Snapshots` are the serializied version of a `Cluster`. The Cluster will be converted into one single blob and send to `Shiori`, where it is written to disc and registered in the database. At the moment only `Cluster-Snapshots`, so at some points in the current implementation it is named als `Snapshot`, which is the same for now. In later versions, there should also exist `Segment-Snapshots`, which are the serialized version of a single segment.

### **Segment-Template** or **Template**

`Segment-Templates` are json-formated strings, which describe the structure of a single type of segment. It can be stored in Kyouko and used to created `Cluster`. 

Originally there was only one type of `Templates` at the beginning, but this didn't scales very well, so it was split later into `Segment-Templates` and `Cluster-Templates`. Because only `Segment-Templates` can be stored, at some parts and also the dashboard when it comes to `Templates`, then it meens `Segment-Templates`.

### **Cluster-Template**

Similar to `Segment-Templates` the `Cluster-Templates` are a json-formated string, which defines the structure of the `Cluster`, by defining which `Segment-Templates` should be used and connected in which way to create the desired `Cluster`.

### **Request-Result** or **Result**

`Request-Results` are the output of a `Request-Task`, which are stored in `Shiori`. They have the same name like the `Request-Task`, which genrated the output.
