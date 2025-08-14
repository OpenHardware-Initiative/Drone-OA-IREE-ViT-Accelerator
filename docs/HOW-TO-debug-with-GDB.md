# Small guide on how I debug with GDB



```bash    
gdb --args build-debug/runtime/plugins/ita-samples/inference/ITAViTLSTM_test_data output/ITAViTLSTM_f16_HOST.vmfb training/small_data
```
  

Inside GDB, set the directory for debug files. The error message shows that the files are inside your build directory, so you just point GDB there.
code Gdb
```bash
(gdb) set debug-file-directory build-debug
```

Press Enter. GDB will now know where to search for the .dwo files.

Now, run your program:
```bash
(gdb) run
```
```bash 
    (gdb) bt
```
      
What you will see now is a much more useful backtrace. Instead of showing cryptic addresses or just function names, it will point to the exact file (main.cpp) and line number where the error occurred. This will allow you to see precisely which line in your print_output_tensor or main function triggered the failure.