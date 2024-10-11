# Cluster Templates

## General

The cluster-templates are used to define the basic structure of a network. This defined structure is
comparable to a plant trellis. It is basically structures in hexagons, which are called hexagons.

![step1](cluster_template_general.drawio)

General structure of the file:

```
version: 1
settings:
    neuron_cooldown: <COOL_DOWN>
    refractory_time: <REFRACTORY_TIME>
    max_connection_distance: <MAX_DISTANCE>
    enable_reduction: <ENABLE_REDUCTION>

hexagons:
    X,Y,Z
    X,Y,Z
    ...
    X,Y,Z

axons:
    X,Y,Z -> X,Y,Z

inputs:
    <NAME>: X,Y,Z
    <NAME>: X,Y,Z (binary)
    <NAME>: X,Y,Z (match)

outputs:
    <NAME>: X,Y,Z
```

In the template the position, type and name of the hexagons has to be defined. The size of the input
and output is defined by the given data when starting a training.

### version

At the moment this is only a placeholder an the `version: 1` is the only supported value at the
moment. There are still too much changes to handle multiple versions currently.

### settings

The entries within this block are all optional. If not listed, the default is used.

-   **neuron_cooldown**

    It the value of how much the potential of a neuron is reduced from one cycle to the next. As
    default it is so high, that the remaining potential has no impact on the next cycle anymore.

-   **refractory_time**

    Gives the number of cycles until a triggered neuron can be triggered again by the input. Within
    this time-perios, only the cooldown of the neuron is active. (Default = 1; Minimum = 1)

-   **max_connection_distance**

    Maximum distance in number-of-hexagon, which a synapses can reach from the source- to the
    target-neuron. (Default = 1; Minimum = 1)

-   **enable_reduction**

    Enable reduction-process to cleanup network. (Defaul: false)

### position

`X,Y,Z` the x-, y- and z-coordinates of the hexagon. All hexagons must be connected by at
least one side with another hexagon.

!!! info

    It is theoretically possible to order them 3-dimensional by using different z-values, but this was never tested until now.

### target

Inputs and outputs is also a `X,Y,Z` and has to be the same position, like the hexagon, where
this input or output should be connected to.

### name

`<NAME>` of this input and output for identification to be able to add input- and output-values.

### binary input (optional setting)

For the case that the input-data have only value 0 and 1, a `(binary)` has to be added at the end of the input:

```
inputs:
    <NAME>: X,Y,Z (binary)
```

Otherwise the results for binary input become really bad. Even the input on the hexagon with this flag is not a binary input, all input-values greater than 0 are automatically handled as 1. 

### matchint input (optional setting)

Use the `(match)` to use the match-mode for inputs.

```
inputs:
    <NAME>: X,Y,Z (match)
```

When this is set, for each input, which is not explizitly learned by the hexgon, a new synapse is created.

### axons (optional)

The `axons`-section is not required and allow to connect any hexagon within the cluster with another one. 

## Simple example

The following is a minimal example for a cluster-template.

```
version: 1
settings:
    neuron_cooldown: 100000000000.0
    refractory_time: 1
    max_connection_distance: 1
    enable_reduction: false

hexagons:
    1,1,1
    3,1,1
    4,1,1

axons:
    1,1,1 -> 3,1,1

inputs:
    input_hexagon: 1,1,1

outputs:
    output_hexagon: 4,1,1
```

It defines 3 hexagons. It contains an input-hexagon with the name `test_input` and an output-hexagon
with name `test_output`. Based on their position, they are all in a straight line, like in the image
below:

![step1](cluster_template_example.drawio)
