# Documentation Docker Image

For the docker image to be able to build correctly we need do download the sysroot from Kria.

```bash
wget -c -O kria-sysroot.tar.xz https://people.canonical.com/~platform/images/xilinx/kria24-ubuntu-22.04/iot-limerick-kria-classic-server-2204-classic-22.04-kd05-20240223-170-sysroot.tar.xz
```

Be sure to run this command from the `./docker` folder