use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};
use std::time::Duration;

fn main() -> Result<(), anyhow::Error> {
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

    Ok(())
}

