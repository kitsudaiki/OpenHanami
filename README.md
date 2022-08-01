# Kitsumi-AI

**Important** The whole project is still far away from a useful and productive state and as long version `v1.0.0` is not reached, there can be API-breaking changes anytime.

Kitsumi-AI is in short an AI-as-a-Service. In its core it has a quite dynamic aritifical neuronal network, which doesn't has hard connections between the nodes of the network and doesn't require a layer-structure. 

Some features of the original network concept, like the functionality to work as a spiking neuronal network, were temporary removed, until I improved the initializing process and find some useful use-cases to readd the removed functions.

## Why?

The project started originally with the wish to create something, which is not so restricted in its structure like in classical deep-learning networks. I simply wanted create something more like the real functionality of the brain. It wasn't planed this big from the beginning. I was only to challange myself and to learn more about the mysterious artifical intelligence. After I had the first Proof-of-Concept, I just couldn't stop, because I thought I could make something big of it, which helps me to achive my goals. 

I have no experience with machine learning in real world practical projects. So I had no (and still have not) knowledge, what's the requirements for a good deep-learning network and if my concept could bring some benefits. Because of this I simply started to add everything, which came in my mind. When it was, at least for myself, an interesting idea, which doesn't break anything, then I added it. Some features were only the consequence of another previous feature. 

For example: "Would be nice, if it can run on a server" -> "I need a network-protocol for interaction" -> "REST-API would be good for at least controlling tasks" -> "I need a login, so not everyone has access" -> "Why now allow multiple users" -> "I need policies and so on" -> ...

Like this everything was added to the project and this process is still going on and all you can see in the project is a result of this workflow.

## Dependencies

see [Dependencies](subsites/Dependencies.md)

## Workflow

see [Workflow](subsites/Workflow.md)

## Contact for questions

tobias.anker@kitsunemimi.moe

## License

Most of the project is under `Apache2`-license. Many of the absolut generic basic libraries are under `MIT`-license, to give more freedom the use them in other projects. See the individual repositories.
