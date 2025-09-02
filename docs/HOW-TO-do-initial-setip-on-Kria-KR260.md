# HOW TO perform initial setup for the Kria KR260

This guide covers the first steps for connecting to and configuring a Kria KR260 board after flashing the Ubuntu 22.04 image to the micro-SD card.

## 1. Connect via Serial (COM Port)

To get initial access to the board's terminal, connect via the USB-C port.

1.  Find the device name for the serial connection.
    ```bash
    sudo dmesg | grep tty
    ```
    (Look for an entry like `ttyUSB1`)

2.  Connect using a terminal emulator like `screen` or `putty`.
    ```bash
    # Using screen
    sudo screen /dev/ttyUSB1 115200

    # Using putty
    sudo putty /dev/ttyUSB1 -serial -sercfg 115200,8,n,1,N
    ```

3.  The default username is `ubuntu`.

## 2. Find the IP Address

Once logged in via the serial console, find the board's IP address to enable SSH access.

1.  Ensure the Kria is connected to your network via the Ethernet port.
2.  Run the following command on the Kria's terminal:
    ```bash
    ip address show eth0
    ```
    Note the `inet` address (e.g., `10.42.0.14`).

## 3. SSH into the Kria

From your host PC, you can now connect to the Kria over the network.

1.  Open a terminal on your host PC.
2.  Connect using the IP address you found.
    ```bash
    ssh ubuntu@<kria_ip_address>
    ```

## 4. Recovering an Interrupted Script

If a setup script is interrupted (e.g., with Ctrl+C), some packages may be left in a broken state. Run these commands to fix them before re-running your script.

```bash
sudo dpkg --configure -a
sudo apt-get install -f```

## 5. Safe Shutdown

Always perform a clean shutdown before removing power from the board to avoid corrupting the file system.

```bash
sudo shutdown -h now