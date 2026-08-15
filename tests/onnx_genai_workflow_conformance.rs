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
    slot_ids: &[i64],
    request_epochs: &[i64],
    active: &[bool],
    values: &[f32],
    selection: AdapterSelection,
) -> anyhow::Result<PipelineGenerateRequest> {
    let batch = i64::try_from(slot_ids.len())?;
    let mut segments = vec![-1i64; slot_ids.len() * 2];
    let mut adapter_counts = vec![0i64; slot_ids.len()];
    let mut adapter_scales = vec![0.0f32; slot_ids.len() * 2];
    for (row, (&slot_id, &request_epoch)) in slot_ids.iter().zip(request_epochs).enumerate() {
        let identity = onnx_genai_engine::AdapterSlotIdentity {
            slot_id,
            request_epoch,
        };
        if let Some(activations) = selection.rows.get(&identity) {
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
    }
    Ok(PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options: Default::default(),
    })
    .with_input("request.slot_ids", Value::from_slice_i64(slot_ids, &[batch])?)
    .with_input(
        "request.request_epochs",
        Value::from_slice_i64(request_epochs, &[batch])?,
    )
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

#[test]
fn mobius_parameter_adapters_preserve_order_rows_compaction_and_epochs() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&root("adapter")?, EngineConfig::default())?;
    let selection = AdapterSelection::default()
        .with_slot(10, 0, [AdapterActivation::new("red", 1.0)])
        .with_slot(
            20,
            0,
            [AdapterActivation::new("blue", 1.0)],
        )
        .with_slot(
            30,
            0,
            [
                AdapterActivation::new("red", 0.5),
                AdapterActivation::new("blue", 1.0),
            ],
        );
    let output = engine.run_pipeline(adapter_request(
        &[10, 20, 30],
        &[0, 0, 0],
        &[true, false, true],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        selection.clone(),
    )?)?;
    assert_eq!(
        output["result"].to_vec_f32()?,
        vec![2.0, 4.0, 3.0, 4.0, 25.5, 35.0]
    );
    let compacted = engine.run_pipeline(adapter_request(
        &[30, 10],
        &[0, 0],
        &[true, true],
        &[5.0, 6.0, 1.0, 2.0],
        selection,
    )?)?;
    assert_eq!(
        compacted["result"].to_vec_f32()?,
        vec![25.5, 35.0, 2.0, 4.0]
    );
    let reused =
        AdapterSelection::default().with_slot(10, 1, [AdapterActivation::new("blue", 1.0)]);
    let stale = engine.run_pipeline(adapter_request(
        &[10],
        &[1],
        &[true],
        &[1.0, 2.0],
        AdapterSelection::default().with_slot(
            10,
            0,
            [AdapterActivation::new("red", 1.0)],
        ),
    )?)?;
    assert_eq!(stale["result"].to_vec_f32()?, vec![1.0, 2.0]);
    for _ in 0..2 {
        let output = engine.run_pipeline(adapter_request(
            &[10],
            &[1],
            &[true],
            &[1.0, 2.0],
            reused.clone(),
        )?)?;
        assert_eq!(output["result"].to_vec_f32()?, vec![7.0, 10.0]);
    }
    let green =
        AdapterSelection::default().with_slot(40, 0, [AdapterActivation::new("green", 1.0)]);
    let output = engine.run_pipeline(adapter_request(
        &[40],
        &[0],
        &[true],
        &[1.0, 2.0],
        green,
    )?)?;
    assert_eq!(output["result"].to_vec_f32()?, vec![4.0, 5.0]);
    let red =
        AdapterSelection::default().with_slot(50, 0, [AdapterActivation::new("red", 1.0)]);
    for _ in 0..2 {
        let output = engine.run_pipeline(adapter_request(
            &[50],
            &[0],
            &[true],
            &[1.0, 2.0],
            red.clone(),
        )?)?;
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
    let slot_ids = (0..batch).collect::<Vec<_>>();
    decoder_batch_request_with_slots(
        input_ids,
        batch,
        sequence,
        prompt_lengths,
        active,
        &slot_ids,
        max_new_tokens,
    )
}

fn decoder_batch_request_with_slots(
    input_ids: &[i64],
    batch: i64,
    sequence: i64,
    prompt_lengths: &[i64],
    active: &[bool],
    slot_ids: &[i64],
    max_new_tokens: usize,
) -> anyhow::Result<PipelineGenerateRequest> {
    let bool_bytes = active.iter().map(|value| u8::from(*value)).collect();
    let zeros = vec![0_i64; usize::try_from(batch)?];
    let ones = vec![1_i64; usize::try_from(batch)?];
    let negative_ones = vec![-1_i64; usize::try_from(batch)?];
    let floats_zero = vec![0.0_f32; usize::try_from(batch)?];
    let floats_one = vec![1.0_f32; usize::try_from(batch)?];
    assert_eq!(slot_ids.len(), usize::try_from(batch)?);
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
        "package.slot_ids",
        Value::from_slice_i64(slot_ids, &[batch])?,
    )
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
    .with_input("request.seed", Value::from_slice_i64(slot_ids, &[batch])?)
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
        })
        .with_input("package.slot_ids", Value::from_slice_i64(&[0], &[1])?),
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
    let compacted = decoder_batch_request_with_slots(
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
    assert_eq!(compacted_row(0)?, first_tokens);
    assert_eq!(compacted_row(1)?, second_tokens);
    let stable_after = engine
        .execution_island_diagnostics()
        .iter()
        .map(|island| island.stable_binding_runs)
        .sum::<u64>();
    assert!(
        stable_after > stable_before,
        "same-shape row compaction must reuse stable island bindings"
    );

    let inactive = decoder_batch_request(&[4, 5, 6, 0], 2, 2, &[2, 1], &[true, false], 3)?;
    let inactive_output = engine.run_pipeline_outputs(inactive)?;
    let inactive_rows = engine.output_rows_for_role(&inactive_output, WorkflowOutputRole::Tokens);
    assert_eq!(inactive_rows.len(), 1);
    assert_eq!(inactive_rows[0].0, 0);
    assert_eq!(inactive_rows[0].1.to_vec_i64()?, first_tokens);

    let first_inactive = decoder_batch_request(&[4, 5, 6, 0], 2, 2, &[2, 1], &[false, true], 3)?;
    let first_inactive_output = engine.run_pipeline_outputs(first_inactive)?;
    let first_inactive_rows =
        engine.output_rows_for_role(&first_inactive_output, WorkflowOutputRole::Tokens);
    assert_eq!(first_inactive_rows.len(), 1);
    assert_eq!(first_inactive_rows[0].0, 1);
    assert_eq!(first_inactive_rows[0].1.to_vec_i64()?, second_tokens);
    assert_eq!(
        engine
            .structured_output_for_role(&first_inactive_output, WorkflowOutputRole::Tokens)
            .expect("semantic lookup must return the first emitted row")
            .to_vec_i64()?,
        second_tokens
    );

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
    )
    .with_input("package.slot_ids", Value::from_slice_i64(&[0], &[1])?);
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
    .with_input("latent", Value::from_slice_f32(&[1.0; 64], &[1, 4, 4, 4])?);
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

fn tts_request(
    prompt_tokens: &[i64],
    batch: i64,
    slot_ids: &[i64],
) -> anyhow::Result<PipelineGenerateRequest> {
    let rows = usize::try_from(batch)?;
    assert_eq!(prompt_tokens.len(), rows * 2);
    assert_eq!(slot_ids.len(), rows);
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
        )
        .with_input(
            "package.slot_ids",
            Value::from_slice_i64(slot_ids, &[batch])?,
        ),
    )
}

#[test]
fn mobius_tts_workflow_executes_real_producer_graphs() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&root("tts")?, EngineConfig::default())?;
    let output = engine.run_pipeline_outputs(tts_request(&[1, 2], 1, &[0])?)?;
    assert_eq!(output["waveform"].shape()[..2], [1, 1]);
    let first = output["waveform"].to_vec_f32()?;
    assert!(!first.is_empty());

    let mut independent = Engine::from_pipeline_dir(&root("tts")?, EngineConfig::default())?;
    let second_output = independent.run_pipeline_outputs(tts_request(&[3, 4], 1, &[1])?)?;
    let second = second_output["waveform"].to_vec_f32()?;

    let batched = engine.run_pipeline_outputs(tts_request(&[1, 2, 3, 4], 2, &[0, 1])?)?;
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
    let compacted = engine.run_pipeline_outputs(tts_request(&[3, 4, 1, 2], 2, &[1, 0])?)?;
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

    let reused = engine.run_pipeline_outputs(tts_request(&[3, 4], 1, &[0])?)?;
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
    .with_input("serving.slot_ids", Value::from_slice_i64(&[0], &[1])?)
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
