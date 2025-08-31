// --- Target and Layout Definitions ---
#executable_target_embedded_elf_x86_64 = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  cpu = "znver2",
  cpu_features = "+prfchw,-cldemote,+avx,+aes,+sahf,+pclmul,-xop,+crc32,-amx-fp8,+xsaves,-avx512fp16,-usermsr,-sm4,-egpr,+sse4.1,-avx512ifma,+xsave,+sse4.2,-tsxldtrk,-sm3,-ptwrite,-widekl,-movrs,-invpcid,+64bit,+xsavec,-avx10.1-512,-avx512vpopcntdq,+cmov,-avx512vp2intersect,-avx512cd,+movbe,-avxvnniint8,-ccmp,-amx-int8,-kl,-avx10.1-256,-sha512,-avxvnni,-rtm,+adx,+avx2,-hreset,-movdiri,-serialize,-vpclmulqdq,-avx512vl,-uintr,-cf,+clflushopt,-raoint,-cmpccxadd,+bmi,-amx-tile,+sse,-avx10.2-256,-gfni,-avxvnniint16,-amx-fp16,-zu,-ndd,+xsaveopt,+rdrnd,-avx512f,-amx-bf16,-avx512bf16,-avx512vnni,-push2pop2,+cx8,-avx512bw,+sse3,-pku,-nf,-amx-tf32,-amx-avx512,+fsgsbase,+clzero,+mwaitx,-lwp,+lzcnt,+sha,-movdir64b,-ppx,+wbnoinvd,-enqcmd,-amx-transpose,-avx10.2-512,-avxneconvert,-tbm,-pconfig,-amx-complex,+ssse3,+cx16,+bmi2,+fma,+popcnt,-avxifma,+f16c,-avx512bitalg,+rdpru,+clwb,+mmx,+sse2,+rdseed,-avx512vbmi2,-prefetchi,-amx-movrs,+rdpid,-fma4,-avx512vbmi,-shstk,-vaes,-waitpkg,-sgx,+fxsr,-avx512dq,+sse4a",
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128",
  native_vector_size = 32 : index,
  target_triple = "x86_64-unknown-unknown-eabi-elf"
}>

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>

#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module attributes {transform.with_named_sequence} {
  
  // --- Executable Definition ---
  hal.executable private @custom_ita_executable {
    hal.executable.variant public @embedded_elf_x86_64 target(#executable_target_embedded_elf_x86_64) 
    objects([#hal.executable.object<{path = "dummy_dispatch_x86_64.o"}>]) {
      
      // Export for ITAFF (absolute value)
      hal.executable.export public @ITAFF ordinal(0) layout(#pipeline_layout) 
      count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      
      // Export for ITASelfAttention (negation)
      hal.executable.export public @ITASelfAttention ordinal(1) layout(#pipeline_layout)
      count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      
      builtin.module {
        // External function declarations
        func.func private @ITAFF_workgroup(
          memref<?x?x?xf16>, memref<?x?x?xf16>, 
          index, index, index
        ) attributes {hal.import.static}
        
        func.func private @ITASelfAttention_workgroup(
          memref<?x?x?xf16>, memref<?x?x?xf16>, 
          index, index, index
        ) attributes {hal.import.static}
        
        // Wrapper for ITAFF
        func.func @ITAFF() {
          %c0 = arith.constant 0 : index
          
          // Load dimensions from constants
          %d0_i32 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
          %d1_i32 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
          %d2_i32 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
          
          // Cast to index
          %d0 = arith.index_castui %d0_i32 : i32 to index
          %d1 = arith.index_castui %d1_i32 : i32 to index
          %d2 = arith.index_castui %d2_i32 : i32 to index
          
          // Get tensor bindings
          %in_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : 
            memref<?x?x?xf16>{%d0, %d1, %d2}
          %out_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : 
            memref<?x?x?xf16>{%d0, %d1, %d2}
          
          // Call external function
          func.call @ITAFF_workgroup(%in_binding, %out_binding, %d0, %d1, %d2) : 
            (memref<?x?x?xf16>, memref<?x?x?xf16>, index, index, index) -> ()
          return
        }
        
        // Wrapper for ITASelfAttention  
        func.func @ITASelfAttention() {
          %c0 = arith.constant 0 : index
          
          // Load dimensions from constants
          %d0_i32 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
          %d1_i32 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
          %d2_i32 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
          
          // Cast to index
          %d0 = arith.index_castui %d0_i32 : i32 to index
          %d1 = arith.index_castui %d1_i32 : i32 to index
          %d2 = arith.index_castui %d2_i32 : i32 to index
          
          // Get tensor bindings
          %in_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : 
            memref<?x?x?xf16>{%d0, %d1, %d2}
          %out_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : 
            memref<?x?x?xf16>{%d0, %d1, %d2}
          
          // Call external function
          func.call @ITASelfAttention_workgroup(%in_binding, %out_binding, %d0, %d1, %d2) : 
            (memref<?x?x?xf16>, memref<?x?x?xf16>, index, index, index) -> ()
          return
        }
      }
    }
  }

  // --- Utility Functions for Dispatching ---
  util.func private @call_ITAFF(%in_arg: tensor<?x?x?xf16>, %out_arg: tensor<?x?x?xf16>) -> tensor<?x?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %d0 = tensor.dim %in_arg, %c0 : tensor<?x?x?xf16>
    %d1 = tensor.dim %in_arg, %c1 : tensor<?x?x?xf16>
    %d2 = tensor.dim %in_arg, %c2 : tensor<?x?x?xf16>
    
    // Cast dimensions to i32 for constants
    %d0_i32 = arith.index_cast %d0 : index to i32
    %d1_i32 = arith.index_cast %d1 : index to i32
    %d2_i32 = arith.index_cast %d2 : index to i32
    
    %workload = arith.constant 1 : index
    
    // Dispatch with single workgroup - pass both tensors
    %result = flow.dispatch @custom_ita_executable::@embedded_elf_x86_64::@ITAFF[%workload](
      %d0_i32, %d1_i32, %d2_i32, %in_arg, %out_arg
    ) : (i32, i32, i32, tensor<?x?x?xf16>{%d0, %d1, %d2}, tensor<?x?x?xf16>{%d0, %d1, %d2}) 
      -> tensor<?x?x?xf16>{%d0, %d1, %d2}
    
    util.return %result : tensor<?x?x?xf16>
  }
  
  util.func private @call_ITASelfAttention(%in_arg: tensor<?x?x?xf16>, %out_arg: tensor<?x?x?xf16>) -> tensor<?x?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %d0 = tensor.dim %in_arg, %c0 : tensor<?x?x?xf16>
    %d1 = tensor.dim %in_arg, %c1 : tensor<?x?x?xf16>
    %d2 = tensor.dim %in_arg, %c2 : tensor<?x?x?xf16>
    
    // Cast dimensions to i32 for constants
    %d0_i32 = arith.index_cast %d0 : index to i32
    %d1_i32 = arith.index_cast %d1 : index to i32
    %d2_i32 = arith.index_cast %d2 : index to i32
    
    %workload = arith.constant 1 : index
    
    // Dispatch with single workgroup - pass both tensors
    %result = flow.dispatch @custom_ita_executable::@embedded_elf_x86_64::@ITASelfAttention[%workload](
      %d0_i32, %d1_i32, %d2_i32, %in_arg, %out_arg
    ) : (i32, i32, i32, tensor<?x?x?xf16>{%d0, %d1, %d2}, tensor<?x?x?xf16>{%d0, %d1, %d2}) 
      -> tensor<?x?x?xf16>{%d0, %d1, %d2}
    
    util.return %result : tensor<?x?x?xf16>
  }

  // --- Matcher Sequences ---
  transform.named_sequence @match_ITAFF(%root: !transform.any_op {transform.readonly}) 
      -> (!transform.any_value, !transform.any_value) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%arg_in: tensor<?x?x?xf16>, %arg_out: tensor<?x?x?xf16>):
        %abs = linalg.generic {
          indexing_maps = [#map1, #map1], 
          iterator_types = ["parallel", "parallel", "parallel"]
        } ins(%arg_in : tensor<?x?x?xf16>) outs(%arg_out : tensor<?x?x?xf16>) {
          ^bb0(%in: f16, %out: f16):
            %res = math.absf %in : f16
            linalg.yield %res : f16
        } -> tensor<?x?x?xf16>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  transform.named_sequence @match_ITASelfAttention(%root: !transform.any_op {transform.readonly}) 
      -> (!transform.any_value, !transform.any_value) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%arg_in: tensor<?x?x?xf16>, %arg_out: tensor<?x?x?xf16>):
        %neg = linalg.generic {
          indexing_maps = [#map1, #map1], 
          iterator_types = ["parallel", "parallel", "parallel"]
        } ins(%arg_in : tensor<?x?x?xf16>) outs(%arg_out : tensor<?x?x?xf16>) {
          ^bb0(%in: f16, %out: f16):
            %res = arith.negf %in : f16
            linalg.yield %res : f16
        } -> tensor<?x?x?xf16>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  // --- Replacer Sequences ---
  transform.named_sequence @replace_with_ITAFF_call(
      %ins: !transform.any_value {transform.readonly}, 
      %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %executable = transform.util.import_symbol @custom_ita_executable into %module if undefined : 
      (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_ITAFF into %module if undefined : 
      (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
      transform.type_conversion.tensor.cast_shape_dynamic_dims
    } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }
  
  transform.named_sequence @replace_with_ITASelfAttention_call(
      %ins: !transform.any_value {transform.readonly}, 
      %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %executable = transform.util.import_symbol @custom_ita_executable into %module if undefined : 
      (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_ITASelfAttention into %module if undefined : 
      (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
      transform.type_conversion.tensor.cast_shape_dynamic_dims
    } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  // --- Main Transform Entry Point ---
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %funcs = transform.structured.match ops{["func.func", "util.func"]} in %module : 
      (!transform.any_op) -> !transform.any_op
    transform.foreach %funcs : !transform.any_op {
      ^bb1(%func: !transform.any_op):
        // Combine both matches in a single foreach_match to avoid invalidation
        transform.foreach_match in %func
          @match_ITAFF -> @replace_with_ITAFF_call,
          @match_ITASelfAttention -> @replace_with_ITASelfAttention_call
        : (!transform.any_op) -> (!transform.any_op)
    }
    transform.apply_dce to %module : !transform.any_op
    transform.yield
  }
}