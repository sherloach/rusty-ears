use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::ffi::CStr;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

// ugly hack because the callback for new segment is not safe
extern "C" fn whisper_on_segment(
    _ctx: *mut whisper_rs_sys::whisper_context,
    state: *mut whisper_rs_sys::whisper_state,
    _n_new: std::os::raw::c_int,
    _user_data: *mut std::os::raw::c_void,
) {
    let last_segment = unsafe { whisper_rs_sys::whisper_full_n_segments_from_state(state) } - 1;
    let ret =
        unsafe { whisper_rs_sys::whisper_full_get_segment_text_from_state(state, last_segment) };
    if ret.is_null() {
        panic!("Failed to get segment text")
    }
    let c_str = unsafe { CStr::from_ptr(ret) };
    let r_str = c_str.to_str().expect("invalid segment text");
    println!("-> Segment ({}) text: {}", last_segment, r_str)
}

fn main() -> Result<()> {
    // Setup the audio input
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("No input device available");
    let config = device.default_input_config()?;

    println!("Recording for 5 seconds...");

    // Prepare WAV writer
    let spec = hound::WavSpec {
        channels: config.channels() as _,
        sample_rate: config.sample_rate().0 as _,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let writer = Arc::new(Mutex::new(Some(hound::WavWriter::create(
        "output.wav",
        spec,
    )?)));

    // Setup the audio stream
    let writer_2 = Arc::clone(&writer);
    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &_| {
            if let Ok(mut guard) = writer_2.lock() {
                if let Some(writer) = guard.as_mut() {
                    for &sample in data.iter() {
                        let sample = (sample * i16::MAX as f32) as i16;
                        writer.write_sample(sample).unwrap();
                    }
                }
            }
        },
        |err| eprintln!("An error occurred on stream: {}", err),
        None,
    )?;

    // Start recording
    stream.play()?;

    // Record for 5 seconds
    std::thread::sleep(Duration::from_secs(5));

    // Stop recording and save the file
    drop(stream);
    if let Ok(mut guard) = writer.lock() {
        if let Some(writer) = guard.take() {
            writer.finalize()?;
            println!("Recording saved to output.wav");
        }
    }

    // Load a context and model
    let ctx = WhisperContext::new_with_params(
        "/Users/hayden/Documents/programming/funs/audio_recorder/models/GGML-tiny-english.bin",
        WhisperContextParameters::default(),
    )
    .expect("Failed to load model");

    // make a state
    let mut state = ctx.create_state().expect("Failed to create state");

    // create a params object
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_translate(false);
    params.set_language(Some("en"));
    unsafe {
        params.set_new_segment_callback(Some(whisper_on_segment));
    }
    params.set_progress_callback_safe(|progress| println!("Progress: {}", progress));

    let st = std::time::Instant::now();
    // Transcribe audio
    let mut reader = hound::WavReader::open("output.wav").expect("Failed to open WAV file");
    let audio: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

    // run the model
    state.full(params, &audio[..]).expect("failed to run model");

    let num_segments = state
        .full_n_segments()
        .expect("failed to get number of segments");
    for i in 0..num_segments {
        let segment = state
            .full_get_segment_text(i)
            .expect("failed to get segment");
        let start_timestamp = state
            .full_get_segment_t0(i)
            .expect("failed to get start timestamp");
        let end_timestamp = state
            .full_get_segment_t1(i)
            .expect("failed to get end timestamp");
        println!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);
    }

    let et = std::time::Instant::now();

    println!("-> Finished (took {}ms)", (et - st).as_millis());

    Ok(())
}
