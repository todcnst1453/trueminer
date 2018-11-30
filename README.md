# trueminer

> Truechain miner with CUDA support

**Trueminer** is a Truehash GPU mining worker: with trueminer you can mine the coin which relies on the Truehash Proof of Work. This is the actively maintained version of trueminer. It implements the Truehash algorithm, and everyone can integrate it into the mainnet code or apply it as a miner client of the remote agent. See [FAQ](#faq) for more details.

## Features

* Nvidia CUDA mining


## Table of Contents

* [Requirement](#Requirements)
* [Build](#build)
* [Maintainers & Authors](#maintainers--authors)
* [F.A.Q.](#faq)


## Requirements

This project uses [CMake] package manager. I suggest using nvidia GPU GTX1080(GP104). When you have multiple GPUs, ensure you change deviceIdx properly and you should connect your monitor to the GPU which is not used for computing. 

### Common

1. [CMake] >= 3.8
2. [Git](https://git-scm.com/downloads)
3. [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) >= 10.0 

### Windows

[Visual Studio 2015](https://www.visualstudio.com/downloads/);

## Instructions

1. Make sure git submodules are up to date:

2. Create a build directory:

    ```shell
    mkdir build
    cd build
    ```

3. Configure the project with CMake. 

    ```shell
    cmake ..
    ```

4. Build the project.

## Maintainers & Authors

CUDA code is highly dependent on your GPU hardware, you have to change some parameters for different architecture and chip. In most cases, you can change the number of blocks and threads to fillful your machine. 
You can contact me by the email: todcnst1453@gmail.com, Thanks.
