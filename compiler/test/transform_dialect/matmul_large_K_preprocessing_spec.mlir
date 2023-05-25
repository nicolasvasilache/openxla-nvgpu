
// Instructions
// ============
//
// TODO: Improve and connect to lit tests.
//
// Apply transforms as a preprocessing at the graph level.
//
// ```
//   export NVGPU_DIR=${HOME}/github/openxla-nvgpu; \
//   cat ${NVGPU_DIR}/compiler/test/transform_dialect/input_ir/matmul.mlir |\
//   sed "s/\${M}/123/g" | sed "s/\${N}/456/g" | sed "s/\${K}/51234/g" | \
//   sed "s/private @fill_matmul_static(/@fill_matmul_static(/g" | \
//   ${NVGPU_DIR}/build/iree_core/llvm-project/bin/mlir-opt -symbol-dce | \
//   ${NVGPU_DIR}/build/iree_core/tools/iree-compile - \
//     --iree-plugin=openxla-transform --openxla-transform-preprocessing=${NVGPU_DIR}/compiler/test/transform_dialect/matmul_large_K_preprocessing_spec.mlir \
//     --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 \
//     --iree-hal-benchmark-dispatch-repeat-count=5 \
//     --iree-codegen-llvmgpu-use-transform-dialect=${NVGPU_DIR}/compiler/test/transform_dialect/pad_via_pack_pipeline_spec.mlir \
//     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \ 
//     --iree-flow-enable-pad-handling \
//     --mlir-disable-threading
// ```

transform.sequence failures(propagate) {
^bb1(%module_op: !transform.any_op):
  %func = transform.structured.match ops{["func.func"]} in %module_op 
    : (!transform.any_op) -> (!transform.any_op)
  transform.print %func : !transform.any_op

  // Step 1. Generic packing with reduction padding to next multiple of 3456 (i.e. 108 * 32).
  // ========================================================================================
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %module_op
    : (!transform.any_op) -> !transform.any_op
  
  transform.print %matmul : !transform.any_op

  %packed_matmul = transform.structured.pack_greedily %matmul 
      matmul_packed_sizes = [0, 0, 0] 
      matmul_padded_sizes_next_multiple_of = [32, 32, 3456] // 3456 = 108 * 32
      // We want [0, 2, 1] to get back to a mn, mk, kn ordering.
      // Otherwise we'd get mn, mk, nk.
      matmul_inner_dims_order = [0, 2, 1]
    : (!transform.any_op) -> !transform.op<"linalg.generic">

  // Need to apply rank-reducing patterns so that split_reduction only sees a single
  // reduction dimension. Otherwise, split_reduction has multiple options and chokes atm.
  // The upstream fix is simple, add a parameter to specify which of the reductions
  // should be split and use the most minor one as the default.
  transform.iree.apply_patterns %module_op {rank_reducing_linalg} : (!transform.any_op) -> ()

  // %packed_matmul_cast = 
  //   transform.cast %packed_matmul : !transform.op<"linalg.generic"> to !transform.any_op
  %1:4 = transform.structured.split_reduction %packed_matmul 
    { split_factor = 54, insert_split_dimension = 2 } 
      : (!transform.op<"linalg.generic">) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

  // Step 2. Special pack / unpack lowering.
  // ==================================================================
  //
  // To easily make tensor.unpack a noop, rewrite pack/unpack at the graph level.
  // This requires calling iree-compile with --iree-flow-enable-pad-handling.
  // (TODO: drop when possible)
  %pack = transform.structured.match ops{["tensor.pack"]} in %module_op
    : (!transform.any_op) -> !transform.op<"tensor.pack">
  transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">) 
    -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
  %unpack = transform.structured.match ops{["tensor.unpack"]} in %module_op
    : (!transform.any_op) -> !transform.op<"tensor.unpack">
  transform.structured.lower_unpack %unpack : (!transform.op<"tensor.unpack">) 
    -> (!transform.op<"tensor.empty">, 
        !transform.op<"linalg.transpose">,
        !transform.op<"tensor.collapse_shape">,
        !transform.op<"tensor.extract_slice">)
  
  transform.iree.apply_patterns %func { rank_reducing_linalg_via_reshapes } : (!transform.any_op) -> ()
  transform.iree.apply_patterns %func { fold_tensor_subsets } : (!transform.any_op) -> ()
  transform.iree.apply_patterns %func { canonicalization, cse } : (!transform.any_op) -> ()

  transform.print %func : !transform.any_op
}
