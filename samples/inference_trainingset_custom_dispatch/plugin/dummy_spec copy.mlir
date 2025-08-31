// --- Target and Layout Definitions ---
#executable_target_embedded_elf_x86_64 = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  cpu = "znver2",
  cpu_features = "+prfchw,-cldemote,+avx,+aes,+sahf,+pclmul,-xop,+crc32,-amx-fp8,+xsaves,-avx512fp16,-usermsr,-sm4,-egpr,+sse4.1,-avx512ifma,+xsave,+sse4.2,-tsxldtrk,-sm3,-ptwrite,-widekl,-movrs,-invpcid,+64bit,+xsavec,-avx10.1-512,-avx512vpopcntdq,+cmov,-avx512vp2intersect,-avx512cd,+movbe,-avxvnniint8,-ccmp,-amx-int8,-kl,-avx10.1-256,-sha512,-avxvnni,-rtm,+adx,+avx2,-hreset,-movdiri,-serialize,-vpclmulqdq,-avx512vl,-uintr,-cf,+clflushopt,-raoint,-cmpccxadd,+bmi,-amx-tile,+sse,-avx10.2-256,-gfni,-avxvnniint16,-amx-fp16,-zu,-ndd,+xsaveopt,+rdrnd,-avx512f,-amx-bf16,-avx512bf16,-avx512vnni,-push2pop2,+cx8,-avx512bw,+sse3,-pku,-nf,-amx-tf32,-amx-avx512,+fsgsbase,+clzero,+mwaitx,-lwp,+lzcnt,+sha,-movdir64b,-ppx,+wbnoinvd,-enqcmd,-amx-transpose,-avx10.2-512,-avxneconvert,-tbm,-pconfig,-amx-complex,+ssse3,+cx16,+bmi2,+fma,+popcnt,-avxifma,+f16c,-avx512bitalg,+rdpru,+clwb,+mmx,+sse2,+rdseed,-avx512vbmi2,-prefetchi,-amx-movrs,+rdpid,-fma4,-avx512vbmi,-shstk,-vaes,-waitpkg,-sgx,+fxsr,-avx512dq,+sse4a",
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128",
  native_vector_size = 32 : index,
  target_triple = "x86_64-unknown-unknown-eabi-elf"
}>

// Pipeline layout with NO constants, 2 bindings (input and output)
#pipeline_layout = #hal.pipeline.layout<constants = 0, bindings = [
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
        // External function declarations - with dimension parameters
        func.func private @ITAFF_workgroup(
          %in_binding: memref<1x128x128xf16>, 
          %out_binding: memref<1x128x128xf16>,
          %d0: index, %d1: index, %d2: index
        ) attributes {hal.import.static}
        
        func.func private @ITASelfAttention_workgroup(
          %in_binding: memref<1x128x128xf16>, 
          %out_binding: memref<1x128x128xf16>,
          %d0: index, %d1: index, %d2: index
        ) attributes {hal.import.static}
        
        // Wrapper for ITAFF
        func.func @ITAFF() {
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c128 = arith.constant 128 : index
          
          // Get tensor bindings with static dimensions
          %in_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : 
            memref<1x128x128xf16>
          %out_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : 
            memref<1x128x128xf16>
          
          // Call external function with dimensions
          func.call @ITAFF_workgroup(%in_binding, %out_binding, %c1, %c128, %c128) : 
            (memref<1x128x128xf16>, memref<1x128x128xf16>, index, index, index) -> ()
          return
        }
        
        // Wrapper for ITASelfAttention  
        func.func @ITASelfAttention() {
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c128 = arith.constant 128 : index
          
          // Get tensor bindings with static dimensions
          %in_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : 
            memref<1x128x128xf16>
          %out_binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : 
            memref<1x128x128xf16>
          
          // Call external function with dimensions
          func.call @ITASelfAttention_workgroup(%in_binding, %out_binding, %c1, %c128, %c128) : 
            (memref<1x128x128xf16>, memref<1x128x128xf16>, index, index, index) -> ()
          return
        }
      }
    } // hal.executable.variant
  } // hal.executable

  // --- Utility Functions for Dispatching ---
  util.func private @call_ITAFF(%in_arg: tensor<1x128x128xf16>, %out_arg: tensor<1x128x128xf16>) -> tensor<1x128x128xf16> {
    
    %workload = arith.constant 1 : index
    
    // Dispatch with NO constants and only 2 tensors (input and output)
    %result = flow.dispatch @custom_ita_executable::@embedded_elf_x86_64::@ITAFF[%workload](
      %in_arg, %out_arg
    ) : (tensor<1x128x128xf16>, tensor<1x128x128xf16>) 
      -> tensor<1x128x128xf16>
    
    util.return %result : tensor<1x128x128xf16>
  }
  
  util.func private @call_ITASelfAttention(%in_arg: tensor<1x128x128xf16>, %out_arg: tensor<1x128x128xf16>) -> tensor<1x128x128xf16> {
    
    %workload = arith.constant 1 : index
    
    // Dispatch with NO constants and only 2 tensors (input and output)
    %result = flow.dispatch @custom_ita_executable::@embedded_elf_x86_64::@ITASelfAttention[%workload](
      %in_arg, %out_arg
    ) : (tensor<1x128x128xf16>, tensor<1x128x128xf16>) 
      -> tensor<1x128x128xf16>
    
    util.return %result : tensor<1x128x128xf16>
  }

  // --- Matcher Sequences with static shapes ---
  transform.named_sequence @match_ITAFF(%root: !transform.any_op {transform.readonly}) 
      -> (!transform.any_value, !transform.any_value) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%arg_in: tensor<1x128x128xf16>, %arg_out: tensor<1x128x128xf16>):

        %abs = linalg.generic {
          indexing_maps = [#map1, #map1], 
          iterator_types = ["parallel", "parallel", "parallel"]
        } ins(%arg_in : tensor<1x128x128xf16>) outs(%arg_out : tensor<1x128x128xf16>) {
          ^bb0(%in: f16, %out: f16):
            %res = math.absf %in : f16
            linalg.yield %res : f16
        } -> tensor<1x128x128xf16>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  transform.named_sequence @match_ITASelfAttention(%root: !transform.any_op {transform.readonly}) 
      -> (!transform.any_value, !transform.any_value) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%arg_in: tensor<1x128x128xf16>, %arg_out: tensor<1x128x128xf16>):

        %neg = linalg.generic {
          indexing_maps = [#map1, #map1], 
          iterator_types = ["parallel", "parallel", "parallel"]
        } ins(%arg_in : tensor<1x128x128xf16>) outs(%arg_out : tensor<1x128x128xf16>) {
          ^bb0(%in: f16, %out: f16):
            %res = arith.negf %in : f16
            linalg.yield %res : f16
        } -> tensor<1x128x128xf16>
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