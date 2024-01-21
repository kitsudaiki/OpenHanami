# Cluster Templates

## General 

The cluster-templates are used to define the basic structure of a network. This defined structure is comparable to a plant trellis. It is basically structures in hexagons, which are called bricks.

![step1](cluster_template_general.drawio)

General structure of the file:

```
version: 1
settings:
    neuron_cooldown: <COOL_DOWN>
    refractory_time: <REFRACTORY_TIME>
    max_connection_distance: <MAX_DISTANCE>
    enable_reduction: <ENABLE_REDUCTION>
bricks:
    <X>,<Y>,<Z>
        input: <NAME>
        number_of_neurons: <NUMBER_OF_NEURONS>
    <X>,<Y>,<Z>
        number_of_neurons: <NUMBER_OF_NEURONS>
    ...
    <X>,<Y>,<Z>
        output: <NAME>
        number_of_neurons: <NUMBER_OF_NEURONS>
```

In the template the position, type, name and size of the bricks has to be defined. 

### version

At the moment this is only a placeholder an the `version: 1` is the only supported value at the moment. There are still too much changes to handle multiple versions currently.

### settings

The entries within this block are all optional. If not listed, the default is used.

- **neuron_cooldown**

    It the value of how much the potential of a neuron is reduced from one cycle to the next. As default it is so high, that the remaining potential has no impact on the next cycle anymore.

- **refractory_time**

    Gives the number of cycles until a triggered neuron can be triggered again by the input. Within this time-perios, only the cooldown of the neuron is active. (Default = 1; Minimum = 1)

- **max_connection_distance**

    Maximum distance in number-of-brick, which a synapses can reach from the source- to the target-neuron. (Default = 1; Minimum = 1)

- **enable_reduction**

    Enable reduction-process to cleanup network. (Defaul: false)

### position

`<X>,<Y>,<Z>` the x-, y- and z-coordinates of the brick. All bricks must be connected by at least one side with another brick.

### type

`input` and `output` defining the type of the brick. Input-bricks can get input-values, while output-bricks returning values at the end of a run. If noting of them was set, the brick is a normal internal brick, which can not be accessed directly from the outside.

### name

`<NAME>` have to be set to the name, which should identify the brick to place input- and output-values.

### size

`<NUMBER_OF_NEURONS>` has to be replace by the number of neurons of the brick. For input-bricks it is the number of input-values and same for the output-brick. For normal internal bricks the number is only a maximum value, which can be used while learning. So making this insanely high, doesn't improve the quality of the training.

!!! info

    It is theoretically possible to order them 3-dimensional by using different z-values, but this was never tested until now.

## Simple example

The following is a minimal example for a cluster-template.

```
version: 1
settings:
    neuron_cooldown: 100000000000.0
    refractory_time: 1
    max_connection_distance: 1
    enable_reduction: false
bricks:
    1,1,1
        input: test_input
        number_of_neurons: 784
    2,1,1
        number_of_neurons: 400
    3,1,1
        output: test_output
        number_of_neurons: 10
```

It defines 3 bricks. It contains an input-brick with the name `test_input` and an output-brick with name `test_output`. Based on their position, they are all in a straight line, like in the image below:

![step1](cluster_template_example.drawio)


