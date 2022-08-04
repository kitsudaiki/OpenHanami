# Kitsumi-AI

**Important: The whole project is still far away from a useful and productive state and as long version `v1.0.0` is not reached, there can be API-breaking changes anytime.**

Kitsumi-AI is in short an AI-as-a-Service. In its core it has a quite dynamic aritifical neuronal network, which doesn't has hard connections between the nodes of the network and doesn't require a layer-structure. 

Some features of the original network concept, like the functionality to work as a spiking neuronal network, were temporary removed, until I improved the initializing process and find some useful use-cases to readd the removed functions.

## Repository overview and Dependencies

see [Overview and Dependencies](subsites/Dependencies.md)

## Why?

### Why this project?

The project started originally with the wish to create something, which is not so restricted in its structure like in classical deep-learning networks. I simply wanted create something more like the real functionality of the brain. It wasn't planed this big from the beginning. I was only to challange myself and to learn more about the mysterious artifical intelligence. After I had the first Proof-of-Concept, I just couldn't stop, because I thought I could make something big of it, which helps me to achive my goals. 

I have no experience with machine learning in real world practical projects. So I had no (and still have not) knowledge, what's the requirements for a good deep-learning network and if my concept could bring some benefits. Because of this I simply started to add everything, which came in my mind. When it was, at least for myself, an interesting idea, which doesn't break anything, then I added it. Some features were only the consequence of another previous feature. 

For example: "Would be nice, if it can run on a server" -> "I need a network-protocol for interaction" -> "REST-API would be good for at least controlling tasks" -> "I need a login, so not everyone has access" -> "Why now allow multiple users" -> "I need policies and so on" -> ...

Like this everything was added to the project and this process is still going on and all you can see in the project is a result of this workflow. It was not designed from the beginning to what it is now. Whenever I added a new feature, I tried to make it dynamic as possible to not block the project in any direction. 

### Why the name?

`Kitsumi` should be a short form for `Kitsunemimi`, which is the english word for `fox ears`. I like anime fox girls and so I also got as private domain `kitsunemimi.moe`, because for ears are moe. ;) Based on this and because I needed a unique nameing, when I created my first C++-libraries, I decided to name them `libKitsunemimi...`. `Kitsumi` is an easier writable shorter form and sound also cute. Beside this, every letter can stand for another core-component of the current state of the project:

- **K**: Kyouko
- **I**: Izumi
- **T**: Tsugumi/Torii^1
- **S**: Sagiri
- **U**: Azuki^2
- **M**: Misaki
- **I**: Inori

^1 Tsugumi is not really a core-component, but torii is only a small proxy and not a named component, so I decided that they can share the letter `T`
^2 I know `Azuki` doesn't start with `U`, but there was no name with `U`, which I really like and Azuki is to nice for the name, so I don't wanted to give up the name only to be compatible with the `U`. But `A` and `U` are both vocal letters, so they are new enough and this is fine. ;)

## Workflow

see [Workflow](subsites/Workflow.md)

## Contact for questions

tobias.anker@kitsunemimi.moe

## License

Most of the project is under `Apache2`-license. Many of the absolut generic basic libraries are under `MIT`-license, to give more freedom the use them in other projects. See the individual repositories.
