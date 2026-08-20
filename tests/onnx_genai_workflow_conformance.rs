//! Runtime conformance for the checked Mobius workflow packages.
//!
//! This file is copied into `onnx-genai-engine/tests` by Mobius CI so every
//! package is executed by the authoritative ONNX GenAI workflow runtime.

#![allow(clippy::field_reassign_with_default)]

use onnx_genai_engine::{
    AdapterActivation, AdapterSelection, Engine, EngineConfig, GenerateOptions,
    GeneratePrompt, GenerateRequest, PipelineGenerateRequest,
    pipeline::{PipelineEngine, WorkflowOutputRole},
};
use onnx_genai_ort::{DataType, Value};
use std::path::PathBuf;

fn root(name: &str) -> anyhow::Result<PathBuf> {
    let root = std::env::var_os("MOBIUS_WORKFLOW_CONFORMANCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/onnx_genai_workflows")
        });
    Ok(root.join(name))
}

fn options(max_new_tokens: usize) -> GenerateOptions {
    let mut options = GenerateOptions::default();
    options.max_new_tokens = max_new_tokens;
    options.seed = Some(7);
    options
}

fn adapter_request(
    active: &[bool],
    values: &[f32],
    selection: &AdapterSelection,
) -> anyhow::Result<PipelineGenerateRequest> {
    let batch = i64::try_from(selection.rows.len())?;
    let mut segments = vec![-1i64; selection.rows.len() * 2];
    let mut adapter_counts = vec![0i64; selection.rows.len()];
    let mut adapter_scales = vec![0.0f32; selection.rows.len() * 2];
    for (row, activations) in selection.rows.iter().enumerate() {
        adapter_counts[row] = i64::try_from(activations.len())?;
        for (slot, activation) in activations.iter().enumerate() {
            segments[row * 2 + slot] = match activation.adapter.as_str() {
                "blue" => 0,
                "green" => 1,
                "red" => 3,
                other => anyhow::bail!("unknown test adapter {other}"),
            };
            adapter_scales[row * 2 + slot] = activation.scale;
        }
    }
    Ok(PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options: Default::default(),
    })
    .with_input(
        "request.adapter_segments",
        Value::from_slice_i64(&segments, &[batch, 2])?,
    )
    .with_input(
        "request.adapter_counts",
        Value::from_slice_i64(&adapter_counts, &[batch])?,
    )
    .with_input(
        "request.adapter_scales",
        Value::from_slice_f32(&adapter_scales, &[batch, 2])?,
    )
    .with_input(
        "request.active",
        Value::from_raw_bytes(
            active.iter().map(|value| u8::from(*value)).collect(),
            &[batch],
            DataType::Bool,
        )?,
    )
    .with_input(
        "activations",
        Value::from_slice_f32(values, &[batch, 2])?,
    ))
}

/// Adapter composition is positional: the runtime keys its cache by the
/// adapter set a batch row asks for, not by any serialized slot identity. The
/// rows below therefore describe order, per-row composition, and compaction,
/// while the request table that maps a row back to a caller stays private to
/// the runtime.
#[test]
fn mobius_parameter_adapters_preserve_order_rows_and_compaction() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&root("adapter")?, EngineConfig::default())?;
    let selection = AdapterSelection::default()
        .with_row([AdapterActivation::new("red", 1.0)])
        .with_row([AdapterActivation::new("blue", 1.0)])
        .with_row([
            AdapterActivation::new("red", 0.5),
            AdapterActivation::new("blue", 1.0),
        ]);
    let output = engine.run_pipeline(adapter_request(
        &[true, false, true],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &selection,
    )?)?;
    assert_eq!(
        output["result"].to_vec_f32()?,
        vec![2.0, 4.0, 3.0, 4.0, 25.5, 35.0]
    );
    // Compaction reorders the batch: the surviving rows keep their composition
    // while moving to new positions.
    let compacted_selection = AdapterSelection::default()
        .with_row([
            AdapterActivation::new("red", 0.5),
            AdapterActivation::new("blue", 1.0),
        ])
        .with_row([AdapterActivation::new("red", 1.0)]);
    let compacted = engine.run_pipeline(adapter_request(
        &[true, true],
        &[5.0, 6.0, 1.0, 2.0],
        &compacted_selection,
    )?)?;
    assert_eq!(
        compacted["result"].to_vec_f32()?,
        vec![25.5, 35.0, 2.0, 4.0]
    );
    // A row that asks for no adapter is passed through unmodified.
    let unadapted = AdapterSelection::default().with_row([]);
    let bare = engine.run_pipeline(adapter_request(&[true], &[1.0, 2.0], &unadapted)?)?;
    assert_eq!(bare["result"].to_vec_f32()?, vec![1.0, 2.0]);
    let reused = AdapterSelection::default().with_row([AdapterActivation::new("blue", 1.0)]);
    for _ in 0..2 {
        let output = engine.run_pipeline(adapter_request(&[true], &[1.0, 2.0], &reused)?)?;
        assert_eq!(output["result"].to_vec_f32()?, vec![7.0, 10.0]);
    }
    let green = AdapterSelection::default().with_row([AdapterActivation::new("green", 1.0)]);
    let output = engine.run_pipeline(adapter_request(&[true], &[1.0, 2.0], &green)?)?;
    assert_eq!(output["result"].to_vec_f32()?, vec![4.0, 5.0]);
    let red = AdapterSelection::default().with_row([AdapterActivation::new("red", 1.0)]);
    for _ in 0..2 {
        let output = engine.run_pipeline(adapter_request(&[true], &[1.0, 2.0], &red)?)?;
        assert_eq!(output["result"].to_vec_f32()?, vec![2.0, 4.0]);
    }
    let diagnostic = engine.adapter_lifecycle_diagnostic();
    assert_eq!(diagnostic.loads, 4);
    assert!(diagnostic.cache_hits > 0);
    assert_eq!(diagnostic.evictions, 2);
    assert_eq!(diagnostic.reloads, 1);
    assert_eq!(diagnostic.capture_invalidations, 2);
    assert!(diagnostic.replayed_plans > 0);
    Ok(())
}

fn assert_batched_policy_super_island(engine: &PipelineEngine) {
    let diagnostics = engine.execution_island_diagnostics();
    let island = diagnostics
        .iter()
        .find(|island| {
            ["token_sampler", "termination", "token_state_update"]
                .iter()
                .all(|component| island.components.iter().any(|item| item == component))
        })
        .unwrap_or_else(|| {
            panic!(
                "sampler, termination, and state update must share one execution island: \
                 {diagnostics:#?}"
            )
        });
    assert!(island.runs > 0, "batched policy island must execute");
    assert_eq!(island.session_runs, island.runs);
    assert!(
        island.component_boundaries_elided >= 2,
        "the three policy components must execute as a fused island"
    );
    if island.device.starts_with("cuda:") {
        assert_eq!(island.fallback_reason, None);
    } else {
        assert_eq!(
            island.fallback_reason.as_deref(),
            Some("island is not placed on CUDA")
        );
    }
}

fn decoder_batch_request(
    input_ids: &[i64],
    batch: i64,
    sequence: i64,
    prompt_lengths: &[i64],
    active: &[bool],
    max_new_tokens: usize,
) -> anyhow::Result<PipelineGenerateRequest> {
    let seeds = (0..batch).collect::<Vec<_>>();
    decoder_batch_request_with_seeds(
        input_ids,
        batch,
        sequence,
        prompt_lengths,
        active,
        &seeds,
        max_new_tokens,
    )
}

fn decoder_batch_request_with_seeds(
    input_ids: &[i64],
    batch: i64,
    sequence: i64,
    prompt_lengths: &[i64],
    active: &[bool],
    seeds: &[i64],
    max_new_tokens: usize,
) -> anyhow::Result<PipelineGenerateRequest> {
    let bool_bytes = active.iter().map(|value| u8::from(*value)).collect();
    let zeros = vec![0_i64; usize::try_from(batch)?];
    let ones = vec![1_i64; usize::try_from(batch)?];
    let negative_ones = vec![-1_i64; usize::try_from(batch)?];
    let floats_zero = vec![0.0_f32; usize::try_from(batch)?];
    let floats_one = vec![1.0_f32; usize::try_from(batch)?];
    assert_eq!(seeds.len(), usize::try_from(batch)?);
    Ok(PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![0]),
        options: options(max_new_tokens),
    })
    .with_input(
        "request.input_ids",
        Value::from_slice_i64(input_ids, &[batch, sequence])?,
    )
    .with_input(
        "request.prompt_lengths",
        Value::from_slice_i64(prompt_lengths, &[batch])?,
    )
    .with_input(
        "package.active",
        Value::from_raw_bytes(bool_bytes, &[batch], DataType::Bool)?,
    )
    .with_input(
        "package.not_done",
        Value::from_raw_bytes(vec![0; usize::try_from(batch)?], &[batch], DataType::Bool)?,
    )
    .with_input("package.one_token", Value::from_slice_i64(&ones, &[batch])?)
    .with_input(
        "request.eos_ids",
        Value::from_slice_i64(&vec![2_i64; usize::try_from(batch)?], &[batch, 1])?,
    )
    .with_input(
        "request.eos_lengths",
        Value::from_slice_i64(&ones, &[batch])?,
    )
    .with_input(
        "request.row_max_iterations",
        Value::from_slice_i64(&negative_ones, &[batch])?,
    )
    .with_input(
        "request.temperature",
        Value::from_slice_f32(&floats_one, &[batch])?,
    )
    .with_input("request.top_k", Value::from_slice_i64(&ones, &[batch])?)
    .with_input(
        "request.top_p",
        Value::from_slice_f32(&floats_one, &[batch])?,
    )
    .with_input(
        "request.min_p",
        Value::from_slice_f32(&floats_zero, &[batch])?,
    )
    .with_input("request.seed", Value::from_slice_i64(seeds, &[batch])?)
    .with_input(
        "request.rng_counter",
        Value::from_slice_i64(&zeros, &[batch])?,
    )
    .with_input(
        "package.cache_lengths",
        Value::from_slice_i64(&zeros, &[batch])?,
    )
    .with_input(
        "package.zero_batch",
        Value::from_slice_i64(&zeros, &[batch])?,
    ))
}

#[test]
fn mobius_decoder_workflow_executes() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&root("decoder")?, EngineConfig::default())?;
    let output = engine.run_pipeline_outputs(
        PipelineGenerateRequest::new(GenerateRequest {
            prompt: GeneratePrompt::TokenIds(vec![4, 5]),
            options: options(3),
        }),
    )?;
    assert_eq!(
        engine
            .structured_output_for_role(&output, WorkflowOutputRole::Tokens)
            .expect("decoder must emit tokens")
            .to_vec_i64()?
            .len(),
        3
    );
    assert_batched_policy_super_island(&engine);
    Ok(())
}

#[test]
fn mobius_decoder_rows_match_independent_runs_and_dynamic_batch_replay() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&root("decoder")?, EngineConfig::default())?;
    let generated = engine.generate_with_pipeline_request(decoder_batch_request(
        &[4, 5],
        1,
        2,
        &[2],
        &[true],
        3,
    )?)?;
    assert_eq!(generated.token_ids.len(), 3);

    let first = decoder_batch_request(&[4, 5], 1, 2, &[2], &[true], 3)?;
    let first_output = engine.run_pipeline_outputs(first)?;
    let first_tokens = engine
        .structured_output_for_role(&first_output, WorkflowOutputRole::Tokens)
        .expect("batch-one decoder must emit tokens")
        .to_vec_i64()?;

    let multi_row_error = engine
        .generate_with_pipeline_request(decoder_batch_request(
            &[4, 5, 6, 0],
            2,
            2,
            &[2, 1],
            &[true, true],
            3,
        )?)
        .expect_err("generate must not flatten multiple semantic rows");
    assert!(
        multi_row_error
            .to_string()
            .contains("multi-row ragged output"),
        "{multi_row_error:#}"
    );

    let batched = decoder_batch_request(&[4, 5, 6, 0], 2, 2, &[2, 1], &[true, true], 3)?;
    let batched_output = engine.run_pipeline_outputs(batched)?;
    let rows = engine.output_rows_for_role(&batched_output, WorkflowOutputRole::Tokens);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].0, 0);
    assert_eq!(rows[0].1.to_vec_i64()?, first_tokens);

    let mut independent = Engine::from_pipeline_dir(&root("decoder")?, EngineConfig::default())?;
    let second = decoder_batch_request(&[6, 0], 1, 2, &[1], &[true], 3)?;
    let second_output = independent.run_pipeline_outputs(second)?;
    let second_tokens = independent
        .structured_output_for_role(&second_output, WorkflowOutputRole::Tokens)
        .expect("independent second row must emit tokens")
        .to_vec_i64()?;
    assert_eq!(rows[1].0, 1);
    assert_eq!(rows[1].1.to_vec_i64()?, second_tokens);

    let stable_before = engine
        .execution_island_diagnostics()
        .iter()
        .map(|island| island.stable_binding_runs)
        .sum::<u64>();
    let compacted = decoder_batch_request_with_seeds(
        &[6, 0, 4, 5],
        2,
        2,
        &[1, 2],
        &[true, true],
        &[1, 0],
        3,
    )?;
    let compacted_output = engine.run_pipeline_outputs(compacted)?;
    let compacted_rows =
        engine.output_rows_for_role(&compacted_output, WorkflowOutputRole::Tokens);
    let compacted_row = |semantic_id| {
        compacted_rows
            .iter()
            .find(|(row_id, _)| *row_id == semantic_id)
            .expect("compacted semantic row must be present")
            .1
            .to_vec_i64()
    };
    // Semantic row ids are positional: the runtime maps a batch row back to a
    // caller through its own private request table, so the reordered batch
    // reports the sequence it actually placed in each row.
    assert_eq!(compacted_row(0)?, second_tokens);
    assert_eq!(compacted_row(1)?, first_tokens);
    let stable_after = engine
        .execution_island_diagnostics()
        .iter()
        .map(|island| island.stable_binding_runs)
        .sum::<u64>();
    assert!(
        stable_after > stable_before,
        "same-shape row compaction must reuse stable island bindings"
    );

    // This decoder concatenates `present` onto `past`, so its cache is dynamic
    // and its attention mask is a dense carry that grows one column per step
    // for every row at once. A partially active batch is therefore not
    // expressible: preserving an inactive row would require its mask to keep
    // the narrower width the rest of the batch has already outgrown. The
    // runtime says so instead of silently corrupting the held row, and this
    // asserts that contract rather than leaving it to chance.
    for active in [[true, false], [false, true]] {
        let mixed = decoder_batch_request(&[4, 5, 6, 0], 2, 2, &[2, 1], &active, 3)?;
        let Err(error) = engine.run_pipeline_outputs(mixed) else {
            panic!("a growing dense carry cannot hold an inactive row");
        };
        assert!(
            format!("{error:#}").contains("cannot preserve inactive rows"),
            "{error:#}"
        );
    }

    let replay = decoder_batch_request(&[6, 0], 1, 2, &[1], &[true], 3)?;
    let replay_output = engine.run_pipeline_outputs(replay)?;
    assert_eq!(
        engine
            .structured_output_for_role(&replay_output, WorkflowOutputRole::Tokens)
            .expect("batch-one replay must emit tokens")
            .to_vec_i64()?,
        second_tokens
    );
    Ok(())
}

#[test]
fn mobius_vlm_workflow_executes_complete_image_path() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&root("vlm")?, EngineConfig::default())?;
    let png = vec![
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 2,
        0, 0, 0, 144, 119, 83, 222, 0, 0, 0, 12, 73, 68, 65, 84, 120, 156, 99, 248, 207, 192, 0, 0,
        3, 1, 1, 0, 201, 254, 146, 239, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
    ];
    let png_len = i64::try_from(png.len())?;
    let request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![4, 5]),
        options: options(2),
    })
    .with_input(
        "request.image",
        Value::from_raw_bytes(png, &[png_len], DataType::Uint8)?,
    );
    let output = engine.run_pipeline_outputs(request)?;
    assert_eq!(
        engine
            .structured_output_for_role(&output, WorkflowOutputRole::Tokens)
            .expect("VLM must emit tokens")
            .shape(),
        [1, 2]
    );
    assert_batched_policy_super_island(&engine);
    Ok(())
}

#[test]
fn mobius_euler_diffusion_workflow_executes_complete_path() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&root("diffusion")?, EngineConfig::default())?;
    let request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![1, 2]),
        options: options(2),
    })
    .with_input(
        "request.noise",
        Value::from_slice_f32(&[1.0; 64], &[1, 4, 4, 4])?,
    );
    let output = engine.run_pipeline_outputs(request)?;
    assert_eq!(output["image"].shape(), [1, 3, 4, 4]);
    assert!(
        output["image"]
            .to_vec_f32()?
            .iter()
            .all(|value| value.is_finite())
    );
    Ok(())
}

#[test]
fn mobius_masked_diffusion_workflow_executes() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&root("masked")?, EngineConfig::default())?;
    let request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![0, 0]),
        options: options(3),
    })
    .with_input(
        "masked_positions",
        Value::from_raw_bytes(vec![1, 0], &[1, 2], DataType::Bool)?,
    )
    .with_input("rng_offset", Value::from_slice_i64(&[0], &[1])?);
    let output = engine.run_pipeline_outputs(request)?;
    assert_eq!(output["tokens"].shape(), [1, 2]);
    Ok(())
}

#[test]
fn mobius_codec_workflow_executes() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&root("codec")?, EngineConfig::default())?;
    let request =
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])))
            .with_input(
                "request.waveform",
                Value::from_slice_f32(&[0.25, -0.5], &[1, 1, 2])?,
            );
    let output = engine.run_pipeline_outputs(request)?;
    assert_eq!(output["waveform"].to_vec_f32()?, [0.25, -0.5]);
    Ok(())
}

fn tts_request(prompt_tokens: &[i64], batch: i64) -> anyhow::Result<PipelineGenerateRequest> {
    let rows = usize::try_from(batch)?;
    assert_eq!(prompt_tokens.len(), rows * 2);
    Ok(
        PipelineGenerateRequest::new(GenerateRequest {
            prompt: GeneratePrompt::TokenIds(vec![0]),
            options: options(1),
        })
        .with_input(
            "request.prompt_tokens",
            Value::from_slice_i64(prompt_tokens, &[batch, 2])?,
        )
        .with_input(
            "package.false",
            Value::from_raw_bytes(vec![0; rows], &[batch], DataType::Bool)?,
        )
        .with_input(
            "package.zero_batch",
            Value::from_slice_i64(&vec![0; rows], &[batch])?,
        )
        .with_input(
            "package.one_batch",
            Value::from_slice_i64(&vec![1; rows], &[batch])?,
        )
        .with_input(
            "package.true",
            Value::from_raw_bytes(vec![1; rows], &[batch], DataType::Bool)?,
        ),
    )
}

#[test]
fn mobius_tts_workflow_executes_real_producer_graphs() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&root("tts")?, EngineConfig::default())?;
    let output = engine.run_pipeline_outputs(tts_request(&[1, 2], 1)?)?;
    assert_eq!(output["waveform"].shape()[..2], [1, 1]);
    let first = output["waveform"].to_vec_f32()?;
    assert!(!first.is_empty());

    let mut independent = Engine::from_pipeline_dir(&root("tts")?, EngineConfig::default())?;
    let second_output = independent.run_pipeline_outputs(tts_request(&[3, 4], 1)?)?;
    let second = second_output["waveform"].to_vec_f32()?;

    let batched = engine.run_pipeline_outputs(tts_request(&[1, 2, 3, 4], 2)?)?;
    let frames = first.len();
    assert_eq!(batched["waveform"].shape(), [2, 1, i64::try_from(frames)?]);
    let batched_waveform = batched["waveform"].to_vec_f32()?;
    assert_eq!(&batched_waveform[..frames], first);
    assert_eq!(&batched_waveform[frames..], second);

    let stable_before = engine
        .execution_island_diagnostics()
        .iter()
        .map(|island| island.stable_binding_runs)
        .sum::<u64>();
    let compacted = engine.run_pipeline_outputs(tts_request(&[3, 4, 1, 2], 2)?)?;
    let compacted_waveform = compacted["waveform"].to_vec_f32()?;
    assert_eq!(&compacted_waveform[..frames], second);
    assert_eq!(&compacted_waveform[frames..], first);
    let stable_after = engine
        .execution_island_diagnostics()
        .iter()
        .map(|island| island.stable_binding_runs)
        .sum::<u64>();
    assert!(
        stable_after > stable_before,
        "same-shape nested TTS compaction must preserve stable bindings"
    );

    let reused = engine.run_pipeline_outputs(tts_request(&[3, 4], 1)?)?;
    assert_eq!(reused["waveform"].to_vec_f32()?, second);
    Ok(())
}

#[test]
fn mobius_speculative_workflow_executes_rejection_and_correction() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&root("speculative")?, EngineConfig::default())?;
    let request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![1, 2, 3, 4]),
        options: options(1),
    })
    .with_input(
        "verifier.past_key_values.0.key",
        Value::from_slice_f32(&[], &[1, 2, 0, 8])?,
    )
    .with_input("grammar.initial_state", Value::from_slice_i64(&[0], &[1])?)
    .with_input(
        "grammar.transition_table",
        Value::from_slice_i64(&[0; 32], &[1, 32])?,
    )
    .with_input("adaptive.current_k", Value::from_slice_i64(&[4], &[1])?)
    .with_input(
        "adaptive.estimates",
        Value::from_slice_f32(&[0.0; 24], &[1, 24])?,
    )
    .with_input("telemetry.draft_ms", Value::from_slice_f32(&[1.0], &[1])?)
    .with_input("telemetry.target_ms", Value::from_slice_f32(&[1.0], &[1])?);
    let output = engine.run_pipeline_outputs(request)?;
    assert_eq!(output["tokens.row.0"].to_vec_i64()?, [1, 31]);
    Ok(())
}

fn video_request(latent_frames: i64, batch: i64) -> anyhow::Result<PipelineGenerateRequest> {
    let rows = usize::try_from(batch)?;
    // [batch, latent_frames, channels, height, width]. Generating from the flat
    // index keeps row 0 and the leading frames identical across shapes, so the
    // comparisons below isolate the runtime's handling of the temporal axis.
    let elements = batch * latent_frames * 4 * 2 * 2;
    let noise: Vec<f32> = (0..elements)
        .map(|index| (index % 11) as f32 / 11.0 - 0.5)
        .collect();
    Ok(PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options: options(3),
    })
    .with_input(
        "request.noise",
        Value::from_slice_f32(&noise, &[batch, latent_frames, 4, 2, 2])?,
    )
    .with_input(
        "request.encoder_hidden_states",
        Value::from_slice_f32(&vec![0.25; rows * 2 * 32], &[batch, 2, 32])?,
    )
    .with_input(
        "package.false",
        Value::from_raw_bytes(vec![0; rows], &[batch], DataType::Bool)?,
    ))
}

#[test]
fn mobius_video_diffusion_workflow_publishes_causal_temporal_chunks() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&root("video")?, EngineConfig::default())?;

    // Three latent frames decode as a single chunk and expand 2x in time.
    let short = engine.run_pipeline_outputs(video_request(3, 1)?)?;
    assert_eq!(short["video"].shape(), [1, 3, 6, 4, 4]);
    let short_frames = short["video"].to_vec_f32()?;
    assert!(short_frames.iter().all(|value| value.is_finite()));
    let frame = |frames: &[f32], total: usize, channel: usize, time: usize| {
        let start = (channel * total + time) * 16;
        frames[start..start + 16].to_vec()
    };

    // Five latent frames decode as two causal chunks (three frames, then two).
    // The clip is the concatenation along time, and what the first chunk already
    // published must not change once the second one runs: that is what the
    // decoder's carried convolution caches are for.
    let long = engine.run_pipeline_outputs(video_request(5, 1)?)?;
    assert_eq!(long["video"].shape(), [1, 3, 10, 4, 4]);
    let long_frames = long["video"].to_vec_f32()?;
    for channel in 0..3 {
        for time in 0..6 {
            assert_eq!(
                frame(&short_frames, 6, channel, time),
                frame(&long_frames, 10, channel, time),
                "chunk boundary changed already-published frame {time}"
            );
        }
    }

    // A batched request decodes independent clips, and the caches from the
    // previous invocations are gone: row 0 reproduces the single-row clip.
    let batched = engine.run_pipeline_outputs(video_request(3, 2)?)?;
    assert_eq!(batched["video"].shape(), [2, 3, 6, 4, 4]);
    let batched_frames = batched["video"].to_vec_f32()?;
    assert_eq!(&batched_frames[..short_frames.len()], &short_frames[..]);
    Ok(())
}
