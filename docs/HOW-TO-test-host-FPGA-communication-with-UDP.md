    
# HOW TO test Host-to-FPGA communication using Python and UDP

This guide outlines the steps to verify a UDP communication link between a host PC and an FPGA board (like the Kria KR260) running a Python server.

## 1. Set Up SSH Key-Based Authentication for GitHub

If you need to clone repositories onto the FPGA, you must use SSH keys, as GitHub no longer accepts password authentication.

1.  On the FPGA, generate a new SSH key pair:
    ```bash
    ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
    ```
2.  Display the public key:
    ```bash
    cat ~/.ssh/id_rsa.pub
    ```
3.  Copy the output and add it as a new SSH key in your GitHub account settings (`Settings -> SSH and GPG keys -> New SSH key`).

## 2. Establish SSH Connection to FPGA

1.  Use a serial connection or your router's client list to find the FPGA's IP address.
2.  From your host PC, connect to the FPGA:
    ```bash
    ssh <username>@<fpga_ip_address>
    ```

## 3. Transfer Test Scripts to FPGA

Use Secure Copy (`scp`) from your host PC to transfer the necessary Python scripts to the FPGA.

1.  Assume you have a dummy server script (`dummy_fpga_server.py`) and a dependency (`fpga_link.py`).
2.  From your host PC's terminal, run:
    ```bash
    # Create a test directory on the FPGA first via SSH if needed
    # ssh <user>@<ip> "mkdir -p ~/project_test"

    # Copy the files
    scp ./FPGA/dummy_fpga_server.py <user>@<ip>:~/project_test/
    scp ./Host/fpga_link.py <user>@<ip>:~/project_test/
    ```

## 4. Run the Communication Test

1.  **On the FPGA:**
    *   SSH into the FPGA.
    *   Navigate to the test directory: `cd ~/project_test`.
    *   If necessary, edit `dummy_fpga_server.py` to ensure its imports use local paths.
    *   Start the dummy server:
        ```bash
        python3 ./dummy_fpga_server.py
        ```
2.  **On the Host PC:**
    *   Edit your host-side script (`Host/fpga_link.py` in this example) to point to the FPGA's real IP address.
    *   Launch your main simulation or test script that communicates with the FPGA.
        ```bash
        # Example command
        bash launch_evaluation_FPGA.bash 1 vision
        ```

**Result:** The host PC should now send UDP packets to the server running on the FPGA. The server will receive them, process them, and send back responses, completing the communication loop.

