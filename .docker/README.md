# Documentation Docker Image

For the docker image to be able to build correctly we need do download the sysroot from Kria.

```bash
wget -c -O kria-sysroot.tar.xz https://people.canonical.com/~platform/images/xilinx/kria24-ubuntu-22.04/iot-limerick-kria-classic-server-2204-classic-22.04-kd05-20240223-170-sysroot.tar.xz
```

Be sure to run this command from the `./docker` folder

More information for AMD LTS ubuntu images, sysroots, etc. can be found [here](https://ubuntu.com/download/amd).

You can activate the image by calling

```bash
# You should be at root level of the project
cd ..
docker run -it --rm -v "$(pwd):/workspace" kria-cross-compiler
```

Some Tips if you are not familiar with Docker:
- Your image will exit the moment that you stop using it or if you turn off your machine
- If you want more variables or libraries in your image (permanently). I recommend adding them to the `Dockerfile`.