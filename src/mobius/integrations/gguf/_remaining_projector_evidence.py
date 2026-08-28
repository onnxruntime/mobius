# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Immutable evidence records for the remaining GGUF projector cohort.

Artifact records pin complete sidecars, but tests fetch only the bounded header
prefix needed to prove metadata, tensor closure, shapes, and stored qtypes.
Source records bind each graph to the exact llama.cpp implementation and model
configuration used to derive its processor contract.
"""

from __future__ import annotations

from typing import Any

LLAMA_CPP_MMPROJ_SHA = "8d9af256337d1a501250f9bbf4c0859a654bddd6"
BOUNDED_HEADER_BYTES = 16 * 1024**2


REMAINING_MMPROJ_ARTIFACT_RECORDS: tuple[dict[str, Any], ...] = (
    {
        "artifact_id": "cogvlm-chat-v1.1-f16-header",
        "repository": "PandaExpressPatron/cogvlm-chat-gguf",
        "revision": "8076a26b5563f170569805cf17e22401f5c790e8",
        "filename": "mmproj-cogvlm-chat-hf",
        "size": 8_858_987_936,
        "lfs_sha256": "aa2e53f40d8248738fc79f790227fde7f564bce870e496db89b5e90425fa1a4b",
        "bounded_header_bytes": BOUNDED_HEADER_BYTES,
        "bounded_header_sha256": (
            "f06cb0cfd4974db17f9549e87178dbfba65146ecf2529b35c2badb09ed4909b6"
        ),
        "projector_types": ("cogvlm",),
        "paired_text_architecture": "cogvlm",
        "paired_text_target": "cogvlm-13B-chat-v1.1-F16.gguf",
        "metadata": (
            ("clip.vision.embedding_length", 1792),
            ("clip.vision.block_count", 63),
            ("clip.vision.image_size", 490),
            ("clip.vision.patch_size", 14),
            ("clip.vision.projection_dim", 4096),
        ),
        "tensor_qtypes": (("F16", 383), ("F32", 637)),
        "tensor_count": 1020,
        "parity_test": "test_cogvlm_projector_matches_independent_reference",
        "processor_repository": "zai-org/cogvlm-chat-hf",
        "processor_revision": "e29dc3ba206d524bf8efbfc60d80fc4556ab0e3c",
        "processor_files": ("config.json",),
        "processor_class": "CogVLM custom image transform",
        "processor_contract": (
            ("pixel_values", "float32[1,3,490,490]"),
            ("image_features", "1227 rows including BOI/EOI"),
            ("ordering", "BOI, raster-order patch rows, EOI"),
        ),
    },
    {
        "artifact_id": "exaone4-5-33b-f16-header",
        "repository": "LGAI-EXAONE/EXAONE-4.5-33B-GGUF",
        "revision": "0e969634ef24db05151b435970297a6dee634b7e",
        "filename": "mmproj-EXAONE-4.5-33B-F16.gguf",
        "size": 2_574_221_920,
        "lfs_sha256": "98d2ecab5b64edea00314150f276ac4de5816b980dbf591832efc344e0f6295a",
        "bounded_header_bytes": BOUNDED_HEADER_BYTES,
        "bounded_header_sha256": (
            "2aa263be72bbb57706da85fd14515b6e1c323028ee620d68aa8bb7f6d5d1a5f3"
        ),
        "projector_types": ("exaone4_5",),
        "paired_text_architecture": "exaone4",
        "paired_text_target": "EXAONE-4.5-33B-IQ4_XS.gguf",
        "metadata": (
            ("clip.vision.embedding_length", 2048),
            ("clip.vision.attention.head_count", 32),
            ("clip.vision.attention.head_count_kv", 8),
            ("clip.vision.block_count", 28),
            ("clip.vision.n_wa_pattern", 7),
            ("clip.vision.projection_dim", 5120),
        ),
        "tensor_qtypes": (("F16", 144), ("F32", 199)),
        "tensor_count": 343,
        "parity_test": "test_exaone4_5_projector_matches_independent_reference",
        "processor_repository": "LGAI-EXAONE/EXAONE-4.5-33B",
        "processor_revision": "570aa4b15a4f45ba1133072b45f50198f6e3b4fd",
        "processor_files": (
            "config.json",
            "preprocessor_config.json",
            "processor_config.json",
        ),
        "processor_class": "Qwen2VLImageProcessor",
        "processor_contract": (
            ("pixel_values", "float32[total_patches,1176]"),
            ("grid_thw", "int64[num_images,3]"),
            ("pixel_limits", "3136..3211264"),
        ),
    },
    {
        "artifact_id": "hunyuanocr-bf16-header",
        "repository": "ggml-org/HunyuanOCR-GGUF",
        "revision": "8e070c9ad79e4ca97a9b4daa2f1ce17e8759afb1",
        "filename": "mmproj-HunyuanOCR-bf16.gguf",
        "size": 997_235_840,
        "lfs_sha256": "46401739a91d0778d86369bb952db685b215512d61a941c3b859f337f6014fcd",
        "bounded_header_bytes": BOUNDED_HEADER_BYTES,
        "bounded_header_sha256": (
            "15f375226c21c2f069f00abd9eddca211347c93f323bc09df50b859e6b88f086"
        ),
        "projector_types": ("hunyuanvl",),
        "paired_text_architecture": "hunyuan_vl",
        "paired_text_target": "HunyuanOCR-bf16.gguf",
        "metadata": (
            ("clip.vision.embedding_length", 1152),
            ("clip.vision.block_count", 27),
            ("clip.vision.image_size", 2048),
            ("clip.vision.patch_size", 16),
            ("clip.vision.spatial_merge_size", 2),
            ("clip.vision.projection_dim", 1024),
        ),
        "tensor_qtypes": (("BF16", 163), ("F32", 284)),
        "tensor_count": 447,
        "parity_test": "test_hunyuanvl_projector_matches_independent_reference",
        "processor_repository": "tencent/HunyuanOCR",
        "processor_revision": "b7bf72439f11fa076c547edf8777aa85f8e0a027",
        "processor_files": ("config.json", "preprocessor_config.json"),
        "processor_class": "HunYuanVLImageProcessor",
        "processor_contract": (
            ("pixel_values", "float32[1,3,dynamic_height,dynamic_width]"),
            ("position_embeddings", "graph bilinear-resizes v.position_embd.weight"),
            ("ordering", "BOI, rows with image_newline, EOI"),
        ),
    },
    {
        "artifact_id": "janus-pro-1b-f16-header",
        "repository": "mradermacher/Janus-Pro-1B-GGUF",
        "revision": "31ced8c1d0bd842eeb8f27730ba792b16e7cda91",
        "filename": "Janus-Pro-1B.mmproj-f16.gguf",
        "size": 621_824_384,
        "lfs_sha256": "b1f441fa6ef80e9b058808e100afeb65f2a844736b8f4865791b59fad1453b1c",
        "bounded_header_bytes": BOUNDED_HEADER_BYTES,
        "bounded_header_sha256": (
            "039c75d5059a38bfa2274da7c8b1511c4a9d642c84e23cede104beac33f4d226"
        ),
        "projector_types": ("janus_pro",),
        "paired_text_architecture": "llama",
        "paired_text_target": "Janus-Pro-1B.Q8_0.gguf",
        "metadata": (
            ("clip.vision.embedding_length", 1024),
            ("clip.vision.block_count", 24),
            ("clip.vision.image_size", 384),
            ("clip.vision.patch_size", 16),
            ("clip.vision.projection_dim", 2048),
        ),
        "tensor_qtypes": (("F16", 147), ("F32", 246)),
        "tensor_count": 393,
        "parity_test": "test_janus_pro_projector_matches_independent_reference",
        "processor_repository": "deepseek-community/Janus-Pro-1B",
        "processor_revision": "1655280bb75959cc1cb85529a2a8b26e7016072e",
        "processor_files": (
            "config.json",
            "preprocessor_config.json",
            "processor_config.json",
        ),
        "processor_class": "JanusImageProcessor",
        "processor_contract": (
            ("pixel_values", "float32[1,3,384,384]"),
            ("image_features", "float32[576,2048]"),
            ("ordering", "single fixed-size raster grid"),
        ),
    },
    {
        "artifact_id": "kimi-k2-5-f16-header",
        "repository": "AesSedai/Kimi-K2.5-GGUF",
        "revision": "43ea7b530645c4f1d2616ec0d376d92b6e69cc9f",
        "filename": "mmproj-Kimi-K2.5-F16.gguf",
        "size": 952_572_160,
        "lfs_sha256": "9261f190d7b8561fc69f70d2bbbc533d5975704f19c6d2b08fa8ac6c133ec78c",
        "bounded_header_bytes": BOUNDED_HEADER_BYTES,
        "bounded_header_sha256": (
            "3e7c895e916044cadd5faa3b68ab6b023ec2d33ca98013d8d98efa71ec3a5ad2"
        ),
        "projector_types": ("kimik25",),
        "paired_text_architecture": "deepseek2",
        "paired_text_target": "Kimi-K2.5 text GGUF split set",
        "metadata": (
            ("clip.vision.embedding_length", 1152),
            ("clip.vision.block_count", 27),
            ("clip.vision.image_size", 896),
            ("clip.vision.patch_size", 14),
            ("clip.vision.projector.scale_factor", 2),
            ("clip.vision.projection_dim", 7168),
        ),
        "tensor_qtypes": (("F16", 111), ("F32", 224)),
        "tensor_count": 335,
        "parity_test": "test_kimik25_projector_matches_independent_reference",
        "processor_repository": "moonshotai/Kimi-K2.5",
        "processor_revision": "4d01dfe0332d63057c186e0b262165819efb6611",
        "processor_files": (
            "config.json",
            "kimi_k25_processor.py",
            "preprocessor_config.json",
        ),
        "processor_class": "KimiK25VisionProcessor",
        "processor_contract": (
            ("pixel_values", "float32[1,3,dynamic_height,dynamic_width]"),
            ("position_ids", "int64[2,num_patches]"),
            ("pixel_limits", "1568..3211264"),
        ),
    },
    {
        "artifact_id": "kimi-vl-a3b-f16-header",
        "repository": "ggml-org/Kimi-VL-A3B-Thinking-2506-GGUF",
        "revision": "e7dcd093335f922a057772febc7ab27eda985b40",
        "filename": "mmproj-Kimi-VL-A3B-Thinking-2506-f16.gguf",
        "size": 905_371_712,
        "lfs_sha256": "1386854a5031970a92ec458ddcc2dfce575f28edf6463643e50ac2f008068fb3",
        "bounded_header_bytes": BOUNDED_HEADER_BYTES,
        "bounded_header_sha256": (
            "fe078c1ecf7a1b00ebf3e040d54754b3fc32ef0500c75c53c5df29f199ff7289"
        ),
        "projector_types": ("kimivl",),
        "paired_text_architecture": "deepseek2",
        "paired_text_target": "Kimi-VL-A3B-Thinking-2506-Q4_K_M.gguf",
        "metadata": (
            ("clip.vision.embedding_length", 1152),
            ("clip.vision.block_count", 27),
            ("clip.vision.image_size", 896),
            ("clip.vision.patch_size", 14),
            ("clip.vision.projector.scale_factor", 2),
            ("clip.vision.projection_dim", 2048),
        ),
        "tensor_qtypes": (("F16", 165), ("F32", 278)),
        "tensor_count": 443,
        "parity_test": "test_kimivl_projector_matches_independent_reference",
        "processor_repository": "moonshotai/Kimi-VL-A3B-Thinking-2506",
        "processor_revision": "aa1730989e7558695b44ee493623e03bd325a994",
        "processor_files": (
            "config.json",
            "image_processing_kimi_vl.py",
            "preprocessor_config.json",
            "processing_kimi_vl.py",
        ),
        "processor_class": "KimiVLProcessor",
        "processor_contract": (
            ("pixel_values", "float32[1,3,dynamic_height,dynamic_width]"),
            ("position_ids", "int64[2,num_patches]"),
            ("ordering", "dynamic raster grid after 2x2 merge"),
        ),
    },
    {
        "artifact_id": "lfm2-vl-1-6b-f16-header",
        "repository": "LiquidAI/LFM2-VL-1.6B-GGUF",
        "revision": "6121de267003bb4d4f325fe10abdc735aee06747",
        "filename": "mmproj-LFM2-VL-1.6B-F16.gguf",
        "size": 830_339_008,
        "lfs_sha256": "b637bfa6060be2bc7503ec23ba48b407843d08c2ca83f52be206ea8563ccbae2",
        "bounded_header_bytes": BOUNDED_HEADER_BYTES,
        "bounded_header_sha256": (
            "6744dec15d156b007be7a12adbd667a4442ee956f0fcbddbe53e5c195cf00149"
        ),
        "projector_types": ("lfm2",),
        "paired_text_architecture": "lfm2",
        "paired_text_target": "LFM2-VL-1.6B-Q4_0.gguf",
        "metadata": (
            ("clip.vision.embedding_length", 1152),
            ("clip.vision.block_count", 26),
            ("clip.vision.image_size", 256),
            ("clip.vision.patch_size", 16),
            ("clip.vision.projector.scale_factor", 2),
            ("clip.vision.projection_dim", 2048),
        ),
        "tensor_qtypes": (("F16", 159), ("F32", 268)),
        "tensor_count": 427,
        "parity_test": "test_lfm2_projector_matches_independent_reference",
        "processor_repository": "LiquidAI/LFM2.5-VL-1.6B",
        "processor_revision": "919fde3d022e3f90a4716006f993938ee8c2eb97",
        "processor_files": ("config.json", "processor_config.json"),
        "processor_class": "Lfm2VlProcessor",
        "processor_contract": (
            ("pixel_values", "float32[num_tiles,max_patches,768]"),
            ("spatial_shapes", "int64[num_tiles,2]"),
            ("ordering", "detail tiles followed by thumbnail"),
        ),
    },
    {
        "artifact_id": "mimo-v2-5-f16-header",
        "repository": "AesSedai/MiMo-V2.5-GGUF",
        "revision": "eed9c5e5d55c5cf9fba5309e66ad5246ea3ffa13",
        "filename": "mmproj-MiMo-V2.5-F16.gguf",
        "size": 1_458_190_112,
        "lfs_sha256": "e1664b236f6fef3d3c64abb24d4441580cbe0df34333995afc1de8ef96ef5089",
        "bounded_header_bytes": BOUNDED_HEADER_BYTES,
        "bounded_header_sha256": (
            "4e0adf899ed6ad0e485107ce6ecfd38e89da6e688b3006a2787ea3e912f64392"
        ),
        "projector_types": ("mimovl",),
        "paired_text_architecture": "mimo2",
        "paired_text_target": "MiMo-V2.5 text GGUF split set",
        "metadata": (
            ("clip.vision.embedding_length", 1280),
            ("clip.vision.attention.head_count", 32),
            ("clip.vision.attention.head_count_kv", 8),
            ("clip.vision.block_count", 28),
            ("clip.vision.window_size", 64),
            ("clip.vision.projection_dim", 4096),
        ),
        "tensor_qtypes": (("F16", 144), ("F32", 221)),
        "tensor_count": 365,
        "parity_test": "test_mimovl_projector_matches_independent_reference",
        "processor_repository": "XiaomiMiMo/MiMo-V2.5",
        "processor_revision": "63651580ca774f8504f676040460aed3e1244ac1",
        "processor_files": ("config.json", "preprocessor_config.json"),
        "processor_class": "Qwen2VLImageProcessor",
        "processor_contract": (
            ("pixel_values", "float32[total_patches,1536]"),
            ("window_modes", "28 entries in {-1,0,1}"),
            ("row_position_ids", "int64[total_patches,2]"),
            ("column_position_ids", "int64[total_patches,2]"),
            ("window_bias", "float32[total_patches,total_patches]"),
            ("column_indices", "int64[merged_patches]"),
            ("inverse_column_indices", "int64[merged_patches]"),
            (
                "adapter",
                "derive position, window, and column-order tensors from processor grid_thw",
            ),
        ),
    },
    {
        "artifact_id": "minicpm-v4-6-bf16-header",
        "repository": "prithivMLmods/MiniCPM-V-4.6-GGUF",
        "revision": "01a7c0ceb731b733bfe9dc3875dc08004f3596b4",
        "filename": "MiniCPM-V-4.6.mmproj-bf16.gguf",
        "size": 1_110_101_888,
        "lfs_sha256": "7b6ffe05cfbc8afbeec7f1847c865c9b6ca1d78f5e128ff7136536b4a87b67e7",
        "bounded_header_bytes": BOUNDED_HEADER_BYTES,
        "bounded_header_sha256": (
            "964d497bc9097b54837f410bd2b04c069256d1835e2a9a7efcfda082fa2aac60"
        ),
        "projector_types": ("minicpmv4_6",),
        "paired_text_architecture": "qwen35",
        "paired_text_target": "MiniCPM-V-4.6.Q4_K_M.gguf",
        "metadata": (
            ("clip.vision.embedding_length", 1152),
            ("clip.vision.block_count", 27),
            ("clip.vision.image_size", 448),
            ("clip.vision.patch_size", 14),
            ("clip.vision.projector.scale_factor", 4),
            ("clip.vision.projection_dim", 1024),
        ),
        "tensor_qtypes": (("BF16", 170), ("F32", 289)),
        "tensor_count": 459,
        "parity_test": "test_minicpmv4_6_projector_matches_independent_reference",
        "processor_repository": "openbmb/MiniCPM-V-4.6",
        "processor_revision": "36f34a661a4bd35d0dc2294cb044d2584646c7d3",
        "processor_files": ("config.json", "preprocessor_config.json"),
        "processor_class": "MiniCPMV4_6Processor",
        "processor_contract": (
            ("pixel_values", "float32[1,3,14,packed_width]"),
            ("target_sizes", "int32[num_visual_units,2]"),
            ("ordering", "overview and slices in processor order"),
        ),
    },
    {
        "artifact_id": "nemotron-nano-v2-vl-bf16-header",
        "repository": "tomlawrence/NVIDIA-Nemotron-Nano-12B-v2-VL-GGUF",
        "revision": "2b87641797d12e8a316ab509c7752887a6d2660a",
        "filename": "mmproj-BF16.gguf",
        "size": 1_689_151_936,
        "lfs_sha256": "10f5fdb09ae2e122af388b5905c4921e66d8f4cc8382f6d1152fcfe8b3fa229d",
        "bounded_header_bytes": BOUNDED_HEADER_BYTES,
        "bounded_header_sha256": (
            "33c5e9f28a5d7499dd7fc72058df5ee9915a08dda68cf5dc991a3b1d37e8d9d4"
        ),
        "projector_types": ("nemotron_v2_vl",),
        "paired_text_architecture": "nemotron_h",
        "paired_text_target": "NVIDIA-Nemotron-Nano-12B-v2-VL-Q4_0.gguf",
        "metadata": (
            ("clip.vision.embedding_length", 1280),
            ("clip.vision.block_count", 32),
            ("clip.vision.image_size", 512),
            ("clip.vision.patch_size", 16),
            ("clip.vision.projector.scale_factor", 2),
            ("clip.vision.projection_dim", 5120),
        ),
        "tensor_qtypes": (("BF16", 130), ("F32", 260)),
        "tensor_count": 390,
        "parity_test": "test_nemotron_v2_vl_projector_matches_independent_reference",
        "processor_repository": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
        "processor_revision": "ca9543b126e8bf3176916d3d305ccc415f89fd4d",
        "processor_files": (
            "config.json",
            "image_processing.py",
            "preprocessor_config.json",
        ),
        "processor_class": "NemotronNanoVLVisionProcessor",
        "processor_contract": (
            ("pixel_values", "float32[1,3,512,512]"),
            ("register_rows", "16 leading rows removed after ViT"),
            ("image_features", "float32[256,5120]"),
        ),
    },
    {
        "artifact_id": "step3-vl-10b-f16-header",
        "repository": "JamePeng2023/Step3-VL-10B-GGUF",
        "revision": "f0bc308167bb03e463e10ba8dfaa4879092d0e84",
        "filename": "mmproj-Step3-VL-10b-F16.gguf",
        "size": 3_972_829_344,
        "lfs_sha256": "541f08bdb3f4799a31a7d37708ba7990fa173bdb14b233c1485076afe34a08a9",
        "bounded_header_bytes": BOUNDED_HEADER_BYTES,
        "bounded_header_sha256": (
            "eb458306d565cd859d4e3db2cc802c1b92138d80b5b4bc9bb6233cf692648753"
        ),
        "projector_types": ("step3vl",),
        "paired_text_architecture": "qwen3",
        "paired_text_target": "Step3-VL-10B-Q3_K_M.gguf",
        "metadata": (
            ("clip.vision.embedding_length", 1536),
            ("clip.vision.block_count", 47),
            ("clip.vision.image_size", 728),
            ("clip.vision.patch_size", 14),
            ("clip.vision.preproc_image_size", 3024),
            ("clip.vision.projection_dim", 4096),
        ),
        "tensor_qtypes": (("F16", 192), ("F32", 475)),
        "tensor_count": 667,
        "parity_test": "test_step3vl_projector_matches_independent_reference",
        "processor_repository": "stepfun-ai/Step3-VL-10B",
        "processor_revision": "5026053b0c2f5dfaa08fc2d149384162c3c8bca1",
        "processor_files": ("config.json", "processing_step3.py", "processor_config.json"),
        "processor_class": "Step3Processor",
        "processor_contract": (
            ("pixel_values", "float32[1,3,dynamic_height,dynamic_width]"),
            ("position_ids", "int64[2,num_patches]"),
            ("ordering", "detail patches first, overview last"),
        ),
    },
    {
        "artifact_id": "yasa2-reka-edge-f16-header",
        "repository": "Vastined/reka-edge-2603-GGUF",
        "revision": "e6a3fd4d8012aea11c7b0d54b3af020bd2e34366",
        "filename": "mmproj-reka-edge-2603-F16.gguf",
        "size": 1_376_166_016,
        "lfs_sha256": "2671b816c9dc63c075fb41615afb7881a7f5e3e7969c874767573ce5a6cd7bc8",
        "bounded_header_bytes": BOUNDED_HEADER_BYTES,
        "bounded_header_sha256": (
            "fef11ed9c78e6a7a3b79bcfe58678ddef6bfc7c29d3924351d9fdc5e96034ec5"
        ),
        "projector_types": ("yasa2",),
        "paired_text_architecture": "llama",
        "paired_text_target": "reka-edge-2603-Q4_K_M.gguf",
        "metadata": (
            ("clip.vision.embedding_length", 2816),
            ("clip.vision.block_count", 0),
            ("clip.vision.image_size", 512),
            ("clip.vision.patch_size", 4),
            ("clip.vision.projection_dim", 4096),
        ),
        "tensor_qtypes": (("F16", 113), ("F32", 270)),
        "tensor_count": 383,
        "parity_test": "test_yasa2_projector_matches_independent_reference",
        "processor_repository": "RekaAI/reka-edge-2603",
        "processor_revision": "492c81c225fbf5f3263a8245b00827721b119a13",
        "processor_files": (
            "config.json",
            "image_processing_yasa2.py",
            "preprocessor_config.json",
            "processing_yasa2.py",
        ),
        "processor_class": "Yasa2ImageProcessor",
        "processor_contract": (
            ("pixel_values", "float32[1,3,512,512] per tile"),
            ("image_features", "64 rows per tile"),
            ("limitation", "pinned llama.cpp sidecar accepts one tile per invocation"),
        ),
    },
)


REMAINING_MMPROJ_SOURCE_RECORDS: tuple[dict[str, Any], ...] = (
    {
        "evidence_id": "cogvlm-pinned-graph-source",
        "sources": (
            ("ggml-org/llama.cpp", LLAMA_CPP_MMPROJ_SHA, "tools/mtmd/models/cogvlm.cpp"),
            ("ggml-org/llama.cpp", LLAMA_CPP_MMPROJ_SHA, "conversion/cogvlm.py"),
            (
                "zai-org/cogvlm-chat-hf",
                "e29dc3ba206d524bf8efbfc60d80fc4556ab0e3c",
                "config.json",
            ),
        ),
        "finding": "Dedicated fused-QKV CLIP tower, gated projector, and BOI/EOI rows.",
    },
    {
        "evidence_id": "exaone4-5-pinned-graph-source",
        "sources": (
            ("ggml-org/llama.cpp", LLAMA_CPP_MMPROJ_SHA, "tools/mtmd/models/exaone4_5.cpp"),
            ("ggml-org/llama.cpp", LLAMA_CPP_MMPROJ_SHA, "conversion/exaone.py"),
            (
                "LGAI-EXAONE/EXAONE-4.5-33B",
                "570aa4b15a4f45ba1133072b45f50198f6e3b4fd",
                "config.json",
            ),
        ),
        "finding": "Qwen-style vision tower with GQA and exact windowed spatial merger.",
    },
    {
        "evidence_id": "hunyuanvl-pinned-graph-source",
        "sources": (
            ("ggml-org/llama.cpp", LLAMA_CPP_MMPROJ_SHA, "tools/mtmd/models/hunyuanvl.cpp"),
            ("ggml-org/llama.cpp", LLAMA_CPP_MMPROJ_SHA, "conversion/hunyuan.py"),
            ("tencent/HunyuanOCR", "b7bf72439f11fa076c547edf8777aa85f8e0a027", "config.json"),
        ),
        "finding": "Dynamic ViT, convolutional perceiver, row-newline, and boundary topology.",
    },
    {
        "evidence_id": "janus-pro-pinned-graph-source",
        "sources": (
            ("ggml-org/llama.cpp", LLAMA_CPP_MMPROJ_SHA, "tools/mtmd/models/siglip.cpp"),
            ("ggml-org/llama.cpp", LLAMA_CPP_MMPROJ_SHA, "conversion/januspro.py"),
            (
                "deepseek-community/Janus-Pro-1B",
                "1655280bb75959cc1cb85529a2a8b26e7016072e",
                "config.json",
            ),
        ),
        "finding": "Fixed SigLIP tower followed by a two-layer exact-GELU aligner.",
    },
    {
        "evidence_id": "kimik25-pinned-graph-source",
        "sources": (
            ("ggml-org/llama.cpp", LLAMA_CPP_MMPROJ_SHA, "tools/mtmd/models/kimik25.cpp"),
            ("ggml-org/llama.cpp", LLAMA_CPP_MMPROJ_SHA, "conversion/kimivl.py"),
            (
                "moonshotai/Kimi-K2.5",
                "4d01dfe0332d63057c186e0b262165819efb6611",
                "config.json",
            ),
        ),
        "finding": "Bicubic 3D learned positions, converted 2D RoPE, merge, norm, and MLP.",
    },
    {
        "evidence_id": "kimivl-pinned-graph-source",
        "sources": (
            ("ggml-org/llama.cpp", LLAMA_CPP_MMPROJ_SHA, "tools/mtmd/models/kimivl.cpp"),
            ("ggml-org/llama.cpp", LLAMA_CPP_MMPROJ_SHA, "conversion/kimivl.py"),
            (
                "moonshotai/Kimi-VL-A3B-Thinking-2506",
                "aa1730989e7558695b44ee493623e03bd325a994",
                "config.json",
            ),
        ),
        "finding": "Learned 2D RoPE ViT followed by patch merge, LayerNorm, and GELU MLP.",
    },
    {
        "evidence_id": "lfm2-pinned-graph-source",
        "sources": (
            ("ggml-org/llama.cpp", LLAMA_CPP_MMPROJ_SHA, "tools/mtmd/models/siglip.cpp"),
            (
                "LiquidAI/LFM2.5-VL-1.6B",
                "919fde3d022e3f90a4716006f993938ee8c2eb97",
                "config.json",
            ),
        ),
        "finding": "Dynamic SigLIP positions, 2x2 pixel unshuffle, LayerNorm, and GELU MLP.",
    },
    {
        "evidence_id": "meralion-pinned-graph-source",
        "sources": (
            ("ggml-org/llama.cpp", LLAMA_CPP_MMPROJ_SHA, "tools/mtmd/models/whisper-enc.cpp"),
            ("ggml-org/llama.cpp", LLAMA_CPP_MMPROJ_SHA, "conversion/ultravox.py"),
            (
                "MERaLiON/MERaLiON-2-3B",
                "a03e40e9ae4f45fb3d575ed7f67bd9fd5304920d",
                "modeling_meralion2.py",
            ),
        ),
        "finding": "Whisper encoder, stack-15, stacked LayerNorm, and four-linear gated adapter.",
    },
    {
        "evidence_id": "mimovl-pinned-graph-source",
        "sources": (
            ("ggml-org/llama.cpp", LLAMA_CPP_MMPROJ_SHA, "tools/mtmd/models/mimovl.cpp"),
            (
                "XiaomiMiMo/MiMo-V2.5",
                "63651580ca774f8504f676040460aed3e1244ac1",
                "config.json",
            ),
        ),
        "finding": "GQA vision tower with row/column windows, sinks, and F32 down projection.",
    },
    {
        "evidence_id": "minicpmv4-6-pinned-graph-source",
        "sources": (
            ("ggml-org/llama.cpp", LLAMA_CPP_MMPROJ_SHA, "tools/mtmd/models/minicpmv.cpp"),
            (
                "openbmb/MiniCPM-V-4.6",
                "36f34a661a4bd35d0dc2294cb044d2584646c7d3",
                "config.json",
            ),
        ),
        "finding": "Bucketed positions, inserted local-attention merger, and final merger.",
    },
    {
        "evidence_id": "minimax-m3-pinned-graph-source",
        "sources": (
            ("ggml-org/llama.cpp", LLAMA_CPP_MMPROJ_SHA, "tools/mtmd/models/minimax-m3.cpp"),
            ("ggml-org/llama.cpp", LLAMA_CPP_MMPROJ_SHA, "conversion/minimax.py"),
            (
                "MiniMaxAI/MiniMax-M3",
                "f0e1c1e04d40177e4673a22097036854f536e9c0",
                "config.json",
            ),
        ),
        "finding": "Partial two-axis RoPE ViT with two distinct spatial MLP mergers.",
    },
    {
        "evidence_id": "nemotron-v2-vl-pinned-graph-source",
        "sources": (
            (
                "ggml-org/llama.cpp",
                LLAMA_CPP_MMPROJ_SHA,
                "tools/mtmd/models/nemotron-v2-vl.cpp",
            ),
            ("ggml-org/llama.cpp", LLAMA_CPP_MMPROJ_SHA, "conversion/nemotron.py"),
            (
                "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
                "ca9543b126e8bf3176916d3d305ccc415f89fd4d",
                "config.json",
            ),
        ),
        "finding": "RADIO registers, fixed positions, patch merge, RMSNorm, and ReLU-squared MLP.",
    },
    {
        "evidence_id": "step3vl-pinned-graph-source",
        "sources": (
            ("ggml-org/llama.cpp", LLAMA_CPP_MMPROJ_SHA, "tools/mtmd/models/step3vl.cpp"),
            ("ggml-org/llama.cpp", LLAMA_CPP_MMPROJ_SHA, "conversion/step3.py"),
            (
                "stepfun-ai/Step3-VL-10B",
                "5026053b0c2f5dfaa08fc2d149384162c3c8bca1",
                "config.json",
            ),
        ),
        "finding": "Absolute plus axial positions and two convolutional downsamplers.",
    },
    {
        "evidence_id": "yasa2-pinned-graph-source",
        "sources": (
            ("ggml-org/llama.cpp", LLAMA_CPP_MMPROJ_SHA, "tools/mtmd/models/yasa2.cpp"),
            (
                "RekaAI/reka-edge-2603",
                "492c81c225fbf5f3263a8245b00827721b119a13",
                "convert_reka_vlm_to_gguf.py",
            ),
            (
                "RekaAI/reka-edge-2603",
                "492c81c225fbf5f3263a8245b00827721b119a13",
                "config.json",
            ),
        ),
        "finding": "ConvNeXtV2, pre-pool positions, fixed 8x8 pooling, and GELU MLP.",
    },
)
