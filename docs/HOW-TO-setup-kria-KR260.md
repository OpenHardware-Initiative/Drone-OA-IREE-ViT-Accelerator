
# HOW TO setup and test an AXI DMA loopback on Kria KR260

This guide documents the process of implementing and testing an AXI DMA loopback on the Kria KR260. The process involves creating a hardware design in Vivado, controlling it from Ubuntu with a custom Linux kernel module, and verifying the data transfer from the Processing System (PS) to the Programmable Logic (PL) and back.

Recommended resources:
- [Slaying the DMA Dragon on the Kria KR260](https://www.hackster.io/yuricauwerts/slaying-the-dma-dragon-on-the-kria-kr260-a5046e)
- [YouTube Tutorial on DMA](https://www.youtube.com/watch?v=5gA3xnsSrdo&t=647s)

---

## Step 1: Vivado Hardware Design

Create a "loopback" design where data sent from the PS is immediately returned by the PL.

1.  **Create a New Project:** Target the Kria KR260 board.
2.  **Create Block Design:** Instantiate the following IP cores:
    *   Zynq UltraScale+ MPSoC
    *   AXI Direct Memory Access (AXI DMA)
    *   AXI-Stream Data FIFO
    *   Processor System Reset & AXI Interconnect
3.  **Configure the AXI DMA:** Enable both **Read Channel (MM2S)** and **Write Channel (S2MM)**.
4.  **Connect the IPs:**
    *   **Data Path:** Connect `M_AXIS_MM2S` (DMA) -> `S_AXIS` (FIFO), then `M_AXIS` (FIFO) -> `S_AXIS_S2MM` (DMA).
    *   **Control Path:** Connect the AXI-Lite interfaces of the Zynq and AXI DMA to the AXI Interconnect.
    *   **Clocks & Resets:** Connect the necessary clocks and resets from the Zynq.

---

## Step 2: Generate Hardware Artifacts

Generate the necessary files for the Kria's Linux system.

1.  **Generate Bitstream (`.bin`):** In project settings under "Bitstream," check the **`bin_file`** option. Then run Synthesis, Implementation, and "Generate Bitstream."
2.  **Generate Device Tree Overlay (`.dtbo`):**
    *   Export the hardware platform (**File -> Export -> Export Platform**) including the bitstream to create an `.xsa` file.
    *   Open `xsct` and create the device tree source:
        ```bash
        hsi::open_hw_design /path/to/your/design.xsa
        createdts -hw /path/to/your/design.xsa -zocl -platform-name my_dma_platform -git-branch xlnx_rel_v2024.1 -overlay -compile -out ./dtg_output/
        exit
        ```
    *   Compile the source into a binary overlay:
        ```bash
        dtc -I dts -O dtb -o pl.dtbo ./dtg_output/dtg_output/my_dma_platform/psu_cortexa53_0/device_tree_domain/bsp/pl.dtsi
        ```
3.  **Edit the device tree:** Add the following section after the existing DMA configuration. The `dma-names` must match the channel names in the kernel module.
    ```dts
        dmatest_0: dmatest@0 {
                compatible ="xlnx,axidma-chrdev";
                dmas = <&axi_dma_0 0
                    &axi_dma_0 1>;
                dma-names = "axidma0", "axidma1";
            } ;
    };
    ```
4.  **Create `shell.json`:** This file provides metadata for the `xmutil` tool.
    ```json
    {
    "shell_type" : "XRT_FLAT",
    "num_slots" : "1"
    }
    ```

---

## Step 3: Update Kria System Software

A version mismatch between Vivado tools and the Kria's software can cause errors (`OF: ERROR: memory leak...`). Align the versions by updating the Kria's OS.

1.  Connect your Kria to the internet.
2.  Run a full system update:
    ```bash
    sudo apt update
    sudo apt upgrade
    ```
3.  Reboot the board:
    ```bash
    sudo reboot
    ```

---

## Step 4: Deploy and Load the Hardware

1.  **Copy Files to Kria:** Transfer the generated files to the board.
    ```bash
    scp design_1_wrapper.bin pl.dtbo shell.json ubuntu@<kria-ip-address>:~/
    ```
2.  **Load the Application:** SSH into your Kria and run these commands.
    ```bash
    # On the Kria board
    APP_NAME="kr260_dma"
    sudo mkdir -p /lib/firmware/xilinx/${APP_NAME}

    # Copy and rename files
    sudo cp ./shell.json /lib/firmware/xilinx/${APP_NAME}/
    sudo cp ./design_1_wrapper.bin /lib/firmware/xilinx/${APP_NAME}/${APP_NAME}.bit.bin
    sudo cp ./pl.dtbo /lib/firmware/xilinx/${APP_NAME}/${APP_NAME}.dtbo

    # Load the app
    cd /lib/firmware/xilinx/${APP_NAME}
    sudo xmutil unloadapp
    sudo xmutil loadapp ${APP_NAME}
    ```
    A successful load will display: `kr260_dma: loaded to slot 0`.

---

## Step 5: Create and Compile the Kernel Driver

This kernel module will trigger the DMA test.

1.  **Check Driver Binding:** See what the DMA is bound to.
    ```bash
    ls -la /sys/bus/platform/devices/a0000000.dma/driver 2>/dev/null || echo "No driver bound"
    ```
2.  **Unbind Default Driver:** If necessary, unbind the default `xilinx-vdma` driver.
    ```bash
    echo "a0000000.dma" | sudo tee /sys/bus/platform/drivers/xilinx-vdma/unbind
    ```
3.  **Kernel Module Code (`dma_loopback_driver.c`):**
    ```c
    #include <linux/fs.h>
    #include <linux/slab.h>
    #include <linux/uaccess.h>
    #include <linux/kernel.h>
    #include <linux/module.h>
    #include <linux/init.h>
    #include <linux/completion.h>
    #include <linux/dmaengine.h>
    #include <linux/dma-mapping.h>
    #include <linux/cdev.h>
    #include <linux/string.h>
    #include <linux/mutex.h>

    MODULE_LICENSE("GPL");
    MODULE_AUTHOR("Adapted by Gemini");
    MODULE_DESCRIPTION("A DMA loopback test for an AXI DMA with a FIFO.");

    #define DMA_TX_CHANNEL      "dma17chan0" // Memory-to-Device (PS to PL)
    #define DMA_RX_CHANNEL      "dma17chan1" // Device-to-Memory (PL to PS)
    #define BUFFER_SIZE         (1024)

    static u8 *result_buf;
    static DEFINE_MUTEX(loopback_mutex);

    static dev_t my_device_nr;
    static struct class *my_class;
    static struct cdev my_device;
    #define DRIVER_NAME "dma_loopback_driver"
    #define DRIVER_CLASS "dma_loopback_class"

    int run_dma_loopback_test(void);

    static ssize_t driver_read(struct file *filp, char __user *buf, size_t len, loff_t *off) {
        int bytes_to_copy;
        if (*off >= BUFFER_SIZE) { return 0; }
        bytes_to_copy = min(len, (size_t)(BUFFER_SIZE - *off));
        mutex_lock(&loopback_mutex);
        if (copy_to_user(buf, result_buf + *off, bytes_to_copy)) {
            mutex_unlock(&loopback_mutex);
            return -EFAULT;
        }
        mutex_unlock(&loopback_mutex);
        *off += bytes_to_copy;
        return bytes_to_copy;
    }

    static ssize_t driver_write(struct file *File, const char *user_buffer, size_t count, loff_t *offs){
        printk(KERN_INFO "dma_loopback_driver: Write detected, starting DMA loopback test.\n");
        run_dma_loopback_test();
        return count;
    }

    void my_dma_callback(void *param){
        struct completion *cmp = (struct completion *) param;
        complete(cmp);
    }

    static struct file_operations fops = {
        .owner = THIS_MODULE,
        .write = driver_write,
        .read = driver_read,
    };

    static int __init my_init(void) {
        result_buf = kmalloc(BUFFER_SIZE, GFP_KERNEL);
        if (!result_buf) { return -ENOMEM; }
        if (alloc_chrdev_region(&my_device_nr, 0, 1, DRIVER_NAME) < 0) { goto free_result_buf; }
        if ((my_class = class_create(THIS_MODULE, DRIVER_CLASS)) == NULL) { goto unregister_chrdev; }
        if (device_create(my_class, NULL, my_device_nr, NULL, DRIVER_NAME) == NULL) { goto destroy_class; }
        cdev_init(&my_device, &fops);
        if (cdev_add(&my_device, my_device_nr, 1) == -1) { goto destroy_device; }
        pr_info("dma_loopback_driver: Inserted into kernel.\n");
        return 0;

    destroy_device:
        device_destroy(my_class, my_device_nr);
    destroy_class:
        class_destroy(my_class);
    unregister_chrdev:
        unregister_chrdev_region(my_device_nr, 1);
    free_result_buf:
        kfree(result_buf);
        return -1;
    }

    static void __exit my_exit(void) {
        cdev_del(&my_device);
        device_destroy(my_class, my_device_nr);
        class_destroy(my_class);
        unregister_chrdev_region(my_device_nr, 1);
        kfree(result_buf);
        pr_info("dma_loopback_driver: Removed from kernel.\n");
    }

    int run_dma_loopback_test(void) {
        dma_cap_mask_t mask;
        struct dma_chan *tx_chan = NULL, *rx_chan = NULL;
        u8 *tx_buf = NULL, *rx_buf = NULL;
        dma_addr_t tx_dma_addr, rx_dma_addr;
        struct dma_async_tx_descriptor *tx_desc, *rx_desc;
        dma_cookie_t tx_cookie, rx_cookie;
        struct completion tx_cmp, rx_cmp;
        int ret = 0, i, mismatch_found = 0;

        dma_cap_zero(mask);
        dma_cap_set(DMA_SLAVE, mask);
        dma_cap_set(DMA_PRIVATE, mask);

        tx_chan = dma_request_channel(mask, NULL, DMA_TX_CHANNEL);
        if (!tx_chan) { pr_err("dma_loopback_driver: Failed to request TX channel\n"); return -ENODEV; }

        rx_chan = dma_request_channel(mask, NULL, DMA_RX_CHANNEL);
        if (!rx_chan) { pr_err("dma_loopback_driver: Failed to request RX channel\n"); ret = -ENODEV; goto release_channels; }

        tx_buf = dma_alloc_coherent(tx_chan->device->dev, BUFFER_SIZE, &tx_dma_addr, GFP_KERNEL);
        if (!tx_buf) { ret = -ENOMEM; goto release_channels; }

        rx_buf = dma_alloc_coherent(rx_chan->device->dev, BUFFER_SIZE, &rx_dma_addr, GFP_KERNEL);
        if (!rx_buf) { ret = -ENOMEM; goto free_tx_buf; }

        for (i = 0; i < BUFFER_SIZE; i++) { tx_buf[i] = (u8)(i & 0xFF); }
        memset(rx_buf, 0, BUFFER_SIZE);
        
        dma_sync_single_for_device(tx_chan->device->dev, tx_dma_addr, BUFFER_SIZE, DMA_TO_DEVICE);

        rx_desc = dmaengine_prep_slave_single(rx_chan, rx_dma_addr, BUFFER_SIZE, DMA_DEV_TO_MEM, DMA_PREP_INTERRUPT | DMA_CTRL_ACK);
        init_completion(&rx_cmp);
        rx_desc->callback = my_dma_callback;
        rx_desc->callback_param = &rx_cmp;
        rx_cookie = dmaengine_submit(rx_desc);

        tx_desc = dmaengine_prep_slave_single(tx_chan, tx_dma_addr, BUFFER_SIZE, DMA_MEM_TO_DEV, DMA_PREP_INTERRUPT | DMA_CTRL_ACK);
        init_completion(&tx_cmp);
        tx_desc->callback = my_dma_callback;
        tx_desc->callback_param = &tx_cmp;
        tx_cookie = dmaengine_submit(tx_desc);

        dma_async_issue_pending(rx_chan);
        dma_async_issue_pending(tx_chan);

        if (wait_for_completion_timeout(&tx_cmp, msecs_to_jiffies(5000)) == 0) { pr_err("dma_loopback_driver: TX channel timeout!\n"); }
        if (wait_for_completion_timeout(&rx_cmp, msecs_to_jiffies(5000)) == 0) { pr_err("dma_loopback_driver: RX channel timeout!\n"); }

        dma_sync_single_for_cpu(rx_chan->device->dev, rx_dma_addr, BUFFER_SIZE, DMA_FROM_DEVICE);

        for (i = 0; i < BUFFER_SIZE; i++) {
            if (tx_buf[i] != rx_buf[i]) {
                pr_err("dma_loopback_driver: FAILED! Mismatch at byte %d. Expected 0x%02x, got 0x%02x\n", i, tx_buf[i], rx_buf[i]);
                mismatch_found = 1;
                break; 
            }
        }
        if (!mismatch_found) { pr_info("dma_loopback_driver: SUCCESS! Data verification passed.\n"); }
        
        mutex_lock(&loopback_mutex);
        memcpy(result_buf, rx_buf, BUFFER_SIZE);
        mutex_unlock(&loopback_mutex);

    free_rx_buf:
        dma_free_coherent(rx_chan->device->dev, BUFFER_SIZE, rx_buf, rx_dma_addr);
    free_tx_buf:
        dma_free_coherent(tx_chan->device->dev, BUFFER_SIZE, tx_buf, tx_dma_addr);
    release_channels:
        if (rx_chan) dma_release_channel(rx_chan);
        if (tx_chan) dma_release_channel(tx_chan);
        return ret;
    }

    module_init(my_init);
    module_exit(my_exit);
    ```

4.  **Makefile:**
    ```makefile
    obj-m += dma_loopback_driver.o

    all:
        make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

    clean:
        make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
    ```
5.  **Install Kernel Headers:**
    ```bash
    sudo apt install linux-headers-$(uname -r)
    ```
6.  **Compile and Run Test:**
    ```bash
    # Compile the driver
    make

    # Unload any old version
    sudo rmmod dma_loopback_driver

    # Load the new module
    sudo insmod dma_loopback_driver.ko

    # Trigger the DMA test
    echo "start" | sudo tee /dev/dma_loopback_driver

    # Check the kernel log for the result
    sudo dmesg | tail -n 30

    # Read the received data back to a file for inspection
    sudo cat /dev/dma_loopback_driver > received_data.bin
    hexdump -C received_data.bin | less
    ```

---

## Appendix: Creating a Custom AXI Peripheral

1.  In Vivado, navigate to **Tools → Create and Package New IP**.
2.  Choose to create a new AXI peripheral and configure it.
3.  Edit the HDL source code in the designated user-modifiable sections.
4.  Navigate to the **Package IP** tab. If you modified ports, click on **Ports and Interfaces** and merge the changes.
5.  Go to **Review and Package** and click **Re-package IP**.
6.  In your block design, you can now add your custom IP.