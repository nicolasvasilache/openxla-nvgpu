// Instructions; TL;DR
// ===================
//
// ```
//   export IREE_DIR=${HOME}/github/iree; \
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/pack.mlir |\
//   sed "s/\${N}/999/g" | sed "s/\${M}/1999/g" |\
//   sed "s/\${Npad}/1024/g" | sed "s/\${Mpad}/2048/g" |\
//   ${IREE_DIR}/build/tools/iree-opt \
//     --iree-hal-target-backends=cuda \
//     --iree-abi-transformation-pipeline \
//     --iree-flow-transformation-pipeline \
//     --iree-stream-transformation-pipeline \
//     --iree-hal-configuration-pipeline | \
//   ${IREE_DIR}/build/tools/iree-opt \
//      --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))' \
//      --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/pad_via_pack_pipeline_spec.mlir \
//      --iree-codegen-llvmgpu-enable-transform-dialect-jit=false
// ```
//
// To produce PTX:
// ```
//   export IREE_DIR=${HOME}/github/iree;
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples;
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/pack.mlir | \
//   sed "s/\${N}/999/g" | sed "s/\${M}/1999/g" |\
//   sed "s/\${Npad}/1024/g" | sed "s/\${Mpad}/2048/g" |\
//   ${IREE_DIR}/build/tools/iree-compile - \
//     --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 \
//     --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/pad_via_pack_pipeline_spec.mlir \
//     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false
// ```

transform.sequence failures(propagate) {
  ^bb1(%variant_op: !transform.any_op):
    transform.print %variant_op : !transform.any_op

    // // Step 1. Convert pack/unpack to pad/extract_slice.
    // Do not apply this for pad ops.
    // %pack = transform.structured.match ops{["tensor.pack"]} in %variant_op
    //   : (!transform.any_op) -> !transform.op<"tensor.pack">
    // transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">)
    //   -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">,
    //       !transform.op<"linalg.transpose">)
    // transform.print %variant_op {name = "after conversion to pad"} : !transform.any_op

    // // The insert_slice is now visible, fold the flow.tensor.store away with it.
    // transform.iree.apply_patterns %variant_op {fold_flow}
    //   : (!transform.any_op) -> ()
    // transform.print %variant_op {name = "after flow folding"} : !transform.any_op

    // Step 2. Tile and distribute to blocks.
    // ======================================
    %pad = transform.structured.match ops{["tensor.pad"]} in %variant_op
      : (!transform.any_op) -> !transform.any_op
    %forall_l1, %tiled_pad_l1 = transform.structured.tile_to_forall_op %pad
      tile_sizes [64, 64] ( mapping = [#gpu.block<y>, #gpu.block<x>] ) 
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice
      %forall_l1 : (!transform.any_op) -> ()
    transform.iree.apply_patterns %variant_op
      {canonicalization, cse, tiling_canonicalization}
      : (!transform.any_op) -> ()
    transform.print %variant_op {name = "after first-level tiling"} : !transform.any_op

    // Step 3. When the amount of tiling is bigger than the amount of padding, we
    // statically know the if branch is not taken.
    // Otherwise, we'd have to keep the if branch.
    // ==========================================================================
    %if = transform.structured.match ops{["scf.if"]} in %forall_l1
      : (!transform.any_op) -> !transform.any_op
    transform.scf.take_assumed_branch %if take_else_branch
      : (!transform.any_op) -> ()
    transform.iree.apply_patterns %variant_op
      {canonicalization, cse, tiling_canonicalization}
      : (!transform.any_op) -> ()
    transform.print %variant_op {name = "after take_else_branch"} : !transform.any_op

    // Step 4. Tile and distribute to threads.
    // =======================================
    %forall_l2, %tiled_pad_l2 = transform.structured.tile_to_forall_op %tiled_pad_l1
      num_threads [16, 16] ( mapping = [#gpu.thread<y>, #gpu.thread<x>] )
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.apply_patterns %variant_op
      {canonicalization, cse, tiling_canonicalization}
      : (!transform.any_op) -> ()
    transform.print %variant_op {name = "after second-level tiling"} : !transform.any_op

    // Step 5. When the apply masked vectorization, we statically know the if 
    // branch is not taken.
    // ======================================================================
    %if_2 = transform.structured.match ops{["scf.if"]} in %forall_l2
      : (!transform.any_op) -> !transform.any_op
    transform.scf.take_assumed_branch %if_2 take_else_branch
      : (!transform.any_op) -> ()
    %pad_inside = transform.structured.match ops{["tensor.pad"]} in %variant_op
      : (!transform.any_op) -> !transform.any_op
    transform.structured.masked_vectorize %pad_inside vector_sizes [4, 4]
      : !transform.any_op
    transform.print %variant_op {name = "after masked vectorization"} : !transform.any_op

    // Step 6. Rank-reduce and vectorize.
    // ==================================
    // Lower the masks to allow canonicalizations to kick in.
    %func_v = transform.structured.match ops{["func.func"]} in %variant_op
      : (!transform.any_op) -> !transform.any_op
    %func_v_2 = transform.vector.lower_masked_transfers %func_v
      : (!transform.any_op) -> !transform.any_op
    transform.iree.apply_patterns %func_v_2
      { rank_reducing_linalg, rank_reducing_vector } : (!transform.any_op) -> ()
    %func_v_3 = transform.structured.vectorize %func_v_2
      : (!transform.any_op) -> !transform.any_op
    transform.iree.apply_patterns %variant_op
      {canonicalization, cse, tiling_canonicalization}
      : (!transform.any_op) -> ()
    transform.print %func_v_3 {name = "after vectorization"} : !transform.any_op

    // Step 7. Bufferize and drop HAL descriptor from memref ops.
    // ==========================================================
    // Pre-bufferization canonicalizations and cleanups help avoid extra copies.
    transform.iree.apply_patterns %func_v_3 {canonicalization, cse, licm}
      : (!transform.any_op) -> ()
    transform.iree.eliminate_empty_tensors %func_v_3 : (!transform.any_op) -> ()
    %variant_op_bufferized = transform.iree.bufferize { target_gpu } %variant_op
    : (!transform.any_op) -> (!transform.any_op)
    %func_m = transform.structured.match ops{["func.func"]}
    in %variant_op_bufferized : (!transform.any_op) -> !transform.any_op
    transform.iree.erase_hal_descriptor_type_from_memref %func_m
      : (!transform.any_op) -> ()
    transform.print %variant_op_bufferized {name = "after bufferization"} : !transform.any_op

    // Step 8. Post-bufferization mapping workgroup.
    // =============================================
    transform.iree.forall_to_workgroup %func_m: (!transform.any_op) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %func_m
        workgroup_dims = [16, 16, 1] : (!transform.any_op) -> ()
    transform.print %variant_op_bufferized {name = "after mapping"} : !transform.any_op

    // TODO: multi-buffering and async.
    // ================================

    // %func_m_2 = transform.structured.match ops{["func.func"]}
    //  in %variant_op_bufferized : (!transform.any_op) -> !transform.any_op
    %func_m_2 = transform.vector.lower_masks %func_m
        : (!transform.any_op) -> !transform.any_op
    %func_m_3 = transform.vector.materialize_masks %func_m_2
        : (!transform.any_op) -> !transform.any_op
  
    transform.iree.apply_patterns %func_m_3 { fold_memref_aliases }
      : (!transform.any_op) -> ()
    // Late canonicalizations and cleanups.
    transform.iree.apply_patterns %func_m_3
      {canonicalization, cse, licm, tiling_canonicalization}
      : (!transform.any_op) -> ()
}
