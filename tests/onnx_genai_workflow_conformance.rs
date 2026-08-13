//! Runtime conformance for the checked Mobius workflow packages.
//!
//! This file is copied into `onnx-genai-engine/tests` by Mobius CI so every
//! package is executed by the authoritative ONNX GenAI workflow runtime.

use onnx_genai_engine::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest, PipelineGenerateRequest,
};
use onnx_genai_ort::{DataType, Value};
use std::path::PathBuf;

fn root(name: &str) -> anyhow::Result<PathBuf> {
    let root = std::env::var_os("MOBIUS_WORKFLOW_CONFORMANCE_DIR")
        .ok_or_else(|| anyhow::anyhow!("MOBIUS_WORKFLOW_CONFORMANCE_DIR must be set"))?;
    Ok(PathBuf::from(root).join(name))
}

fn options(max_new_tokens: usize) -> GenerateOptions {
    let mut options = GenerateOptions::default();
    options.max_new_tokens = max_new_tokens;
    options.seed = Some(7);
    options
}

#[test]
fn mobius_decoder_workflow_executes() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&root("decoder")?, EngineConfig::default())?;
    let output = engine.run_pipeline(PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![4, 5]),
        options: options(3),
    }))?;
    assert_eq!(output["tokens"].to_vec_i64()?.len(), 3);
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
    let output = engine.run_pipeline(request)?;
    assert_eq!(output["tokens"].shape(), [1, 2]);
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
    let output = engine.run_pipeline(request)?;
    assert_eq!(output["image"].shape(), [1, 3, 4, 4]);
    assert!(output["image"]
        .to_vec_f32()?
        .iter()
        .all(|value| value.is_finite()));
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
    let output = engine.run_pipeline(request)?;
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
    let output = engine.run_pipeline(request)?;
    assert_eq!(output["waveform"].to_vec_f32()?, [0.25, -0.5]);
    Ok(())
}

#[test]
fn mobius_tts_workflow_executes_real_producer_graphs() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&root("tts")?, EngineConfig::default())?;
    let output = engine.run_pipeline(PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![1, 2]),
        options: options(1),
    }))?;
    assert_eq!(output["waveform"].shape()[..2], [1, 1]);
    assert!(!output["waveform"].to_vec_f32()?.is_empty());
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
    let output = engine.run_pipeline(request)?;
    assert_eq!(output["tokens.row.0"].to_vec_i64()?, [1, 31]);
    Ok(())
}
