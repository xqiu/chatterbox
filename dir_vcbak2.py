import os
import subprocess
import argparse
import sys
import shutil
import re
import datetime
import math
import shlex
from pydub import AudioSegment

import torch
import torchaudio as ta

from chatterbox.vc import ChatterboxVC

RVC_PATH = os.path.dirname(os.path.realpath(__file__))
LOG_FILE = os.path.join(RVC_PATH, "run_infer.log")
log_file = None

#check _reference_duration_vs_process-time.csv to set these values
MIN_SEGMENT_DURATION = 45  # to avoid too short segments; soft limatation (if conflict with max_segment_duration this value wont be used)
MAX_SEGMENT_DURATION = 60  # to avoid too long segments; hard limitation

PADDING_SECONDS = 10
PADDING_HEADING_SECONDS = 9.9

def log_message(message, level="INFO", printthis: bool = True):
    global log_file
    if log_file is None:
        log_file = LOG_FILE
        print(f"[WARNING] using default log file: {log_file}")
    with open(log_file, "a", encoding='utf-8') as log_fd:
        log_fd.write(f"[{level}] {message}\n")
    if printthis:
        safe_print(f"[{level}] {message}")
        
def safe_print(message):
    """Prints a message to the console, ensuring it is safe for all environments."""
    try:
        print(message)
    except UnicodeEncodeError:
        sys.stdout.reconfigure(encoding='utf-8')
        # If there's an encoding error, replace problematic characters
        print(message.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))

def get_audio_detail(input_file):
    """_summary_

    Args:
        input_file (_str_): path of audio file (wav or mp3)

    Returns:
        tuple: (duration: float, sample_rate: int, channel_str: str)
        duration: unit second
        sample_rate: unit Hz
        channel_str: mono, stereo, 5.1, 7.1, unknown
    """
    #using ffprobe to get the duration of the audio file
    detail_cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 
        # ffprobe use it's own order. even you write duration,sample_rate,channels, it still return data in the order of sample_rate, channels, duration
        'stream=sample_rate,channels,duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', input_file
    ]
    try:
        output = subprocess.run(detail_cmd, stdout=subprocess.PIPE, text=True, check=True, encoding='utf-8').stdout.strip()
        sample_rate, channel, duration = output.split('\n')

        #set channel str
        # mono if 1
        # stereo if 2
        # 5.1 if 6
        # 7.1 if 8
        channel_str = 'mono' if channel == '1' else 'stereo' if channel == '2' else '5.1' if channel == '6' else '7.1' if channel == '8' else 'unknown'

        log_message(f"get_audio_detail {input_file} => \n\tduration: {duration}, \n\tsample_rate: {sample_rate}, \n\tchannel: {channel}")

        return float(duration), int(sample_rate), channel_str
    except (subprocess.CalledProcessError, ValueError) as e:
        return -1.0, -1, 'unknown'

def append_silence(input_file: str, append_duration: float, sample_rate: int, channel: str, continue_job, output_file=None) -> str:
    """
    Append silence to the end of the input audio file.

    Args:
        input_file (str): input audio file path
        append_duration (float): duration of silence to append in seconds
        sample_rate (int): sample rate of the audio file
        channel (str): channel of the audio file
    Returns:
        str: output file path (if fail will return same path as input_file)
    """
    log_message(f'append silence to {input_file} by {append_duration} seconds')
    # output file name = input file name appended with '_append'
    if output_file is None:
        output_file = input_file
        if '.wav' in input_file:
            output_file = input_file.rsplit('.wav', 1)[0] + '_append.wav'
        
    #if output_file already exists, skip it
    if continue_job and os.path.exists(output_file):
        log_message(f"Skipping appending silence to {input_file} because {output_file} already exists", level='WARNING')
        safe_print(f"[WARNING] Skipping appending silence to {input_file} because {output_file} already exists")
        return output_file
    
    log_message(f'append silence to {input_file} by {append_duration} seconds')
    ffmpeg_silence_cmd = [
        'ffmpeg', '-y', '-i', input_file, '-f', 'lavfi', '-t', str(append_duration), '-i', f'anullsrc=channel_layout={channel}:sample_rate={sample_rate}', '-filter_complex', '[0][1]concat=n=2:v=0:a=1', output_file
    ]
    try: 
        subprocess.run(ffmpeg_silence_cmd, check=True, encoding='utf-8', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        log_message(f"[Error] appending silence: {e}")
        return input_file

    return output_file


def append_silence_then_head(
    input_file: str,
    silence_duration: float,
    head_duration: float,
    sample_rate: int,
    channel: str,
    continue_job: bool,
    output_file: str,
) -> str:
    """Append tail padding: 1) silence, then 2) first N seconds of the segment.

    Result: output = input + silence_duration + head(input, head_duration)
    """
    silence_duration = max(0.0, float(silence_duration))
    head_duration = max(0.0, float(head_duration))
    if silence_duration == 0.0 and head_duration == 0.0:
        return input_file

    if continue_job and os.path.exists(output_file):
        log_message(
            f"Skipping tail padding for {input_file} because {output_file} already exists",
            level='WARNING',
        )
        return output_file

    ffmpeg_pad_cmd = [
        'ffmpeg', '-y', '-i', input_file,
        '-filter_complex',
        (
            f"[0:a]asplit=2[a0][a1];"
            f"[a0]asetpts=PTS-STARTPTS[a];"
            f"[a1]atrim=start=0:duration={head_duration},asetpts=PTS-STARTPTS[head];"
            f"anullsrc=channel_layout={channel}:sample_rate={sample_rate},atrim=duration={silence_duration},asetpts=PTS-STARTPTS[sil];"
            f"[a][sil][head]concat=n=3:v=0:a=1[out]"
        ),
        '-map', '[out]',
        output_file,
    ]
    try:
        subprocess.run(ffmpeg_pad_cmd, check=True, encoding='utf-8', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        log_message(f"[Error] tail padding: {e}")
        return input_file

    return output_file

def extract_audio_segment(input_file: str, start_time: float, duration: float, output_file: str, continue_job=False) -> bool:
    """
    Extracts a segment from the input audio file using ffmpeg.

    Args:
        input_file (str): Path to the input audio file (wav or mp3).
        start_time (float): Start time of the segment in seconds.
        duration (float): Duration of the segment in seconds.
        output_file (str): Path to save the extracted segment.
        continue_job (bool): If True, skips processing if output_file already exists.

    Returns:
        bool: True if extraction is successful, False otherwise.
    """
    #if output_file already exists, skip it
    if continue_job and os.path.exists(output_file):
        print(f"Skipping extraction of segment {output_file} because it already exists", file=sys.stdout)
        return True

    ffmpeg_extract_cmd = [
        'ffmpeg', '-y', '-i', input_file, '-ss', str(start_time), '-t', str(duration),
        '-c', 'copy', output_file
    ]
    print(f'RUN extract segment => from: {start_time}; length: {duration}', file=sys.stdout)
    try:
        subprocess.run(ffmpeg_extract_cmd, check=True, encoding='utf-8', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Error] extracting segment: {e}", file=sys.stderr)
        return False

def split_audio_into_segments(input_file: str, temp_dir: str, noise_threshold=-30, silence_duration=1.5, continue_job=False, padding_sec=0.1):
    """
    Splits the input audio file into segments labeled as 'silence' and 'non_silence'
    using ffmpeg's silencedetect filter.
    
    Args:
        input_file (str): Path to the input audio file (wav or mp3).
        temp_dir (str): Directory to store the extracted segments.
        noise_threshold (int): Silence threshold in dB.
        silence_duration (float): Minimum duration of silence to detect in seconds.
        continue_job (bool): If True, skips processing segments that already exist.
        padding_sec (float): Padding duration in seconds to add around non-silence segments.

    Returns:
        tuples: (segment_file, segment_type, start_time, end_time)

    Example:
    noise_threshold, silence_duration, Expected Behavior
    * -40dB, d=0.5	Very aggressive: Cuts on very small gaps, even if there’s low noise.
    * -30dB, d=1	Moderate (your current setting): Good for clean recordings, but can overcut in noisy ones.
    * -25dB, d=2	More tolerant: Best if your audio has background hum or short pauses.
    * -20dB, d=3	Very tolerant: Cuts only on long, obvious silence.
    """
    # Get total duration of the input file.
    duration_cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', input_file
    ]
    log_message(f'RUN get file duration: {input_file}', printthis=False)
    total_duration = float(subprocess.run(duration_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', check=True).stdout.strip())
    log_message(f'BEGIN file => {input_file}', printthis=False)
    log_message(f'\t=> total_duration: {total_duration}', printthis=False)

    # Run ffmpeg silencedetect to get silence intervals.
    silence_cmd = [
        'ffmpeg', '-i', input_file, '-af',
        f'silencedetect=noise={noise_threshold}dB:d={silence_duration}', '-f', 'null', '-'
    ]
    log_message(f'RUN get file silence intervals: {input_file}', printthis=False)
    result = subprocess.run(silence_cmd, stderr=subprocess.PIPE, text=True, check=True, encoding='utf-8')
    
    # Parse the stderr output to get pairs of silence_start and silence_end.
    silence_intervals = []
    current_silence_start = None
    for line in result.stderr.splitlines():
        if 'silence_start' in line:
            match = re.search(r'silence_start: (\d+\.?\d*)', line)
            if match:
                current_silence_start = float(match.group(1))
        if 'silence_end' in line and current_silence_start is not None:
            match = re.search(r'silence_end: (\d+\.?\d*)', line)
            if match:
                silence_end = float(match.group(1))

                real_silence_start = current_silence_start + padding_sec
                if(real_silence_start < 0 ):
                    real_silence_start = 0
                real_silence_end = silence_end - padding_sec
                if(real_silence_end > total_duration):
                    real_silence_end = total_duration

                silence_intervals.append((real_silence_start, real_silence_end))
                current_silence_start = None

    # Create segments: non-silence segments are between silence intervals.
    segments = []  # Each element is (start_time, end_time, segment_type)
    current_time = 0.0
    log_message(f'BEGIN RAW silence_intervals', printthis=True)
    for (silence_start, silence_end) in silence_intervals:
        log_message(f'{silence_start}\t{silence_end}', printthis=False)
        adjust_silence_start = silence_start
        adjust_silence_end = silence_end
        #adjust silence_start and silence_end
        # 1. precision only to 1 decimal place
        # 2. adjust_silence_start will be greater or equal to silence_start
        # 3. adjust_silence_end will be less or equal to silence_end
        # adjust_silence_start = round(silence_start, 1)
        # adjust_silence_end = round(silence_end, 1)
        # if adjust_silence_start < silence_start:
        #     adjust_silence_start += 0.1
        # if adjust_silence_end > silence_end:
        #     adjust_silence_end -= 0.1
        # If there is audio before this silence, add it as a non-silence segment.
        if adjust_silence_start > current_time:
            segments.append((current_time, adjust_silence_start, 'non_silence'))
        # Add the silence segment.
        segments.append((adjust_silence_start, adjust_silence_end, 'silence'))
        current_time = adjust_silence_end
    log_message(f'END RAW silence_intervals', printthis=True)
    # If there is audio after the last silence, add it.
    if current_time < total_duration:
        segments.append((current_time, total_duration, 'non_silence'))

    # 5) MERGE PASS
    # rule: if segment is non_silence and < MIN, keep merging forward (even silence) until >= MIN
    merged_segments = []
    i = 0
    n = len(segments)

    log_message(f'BEGIN segments of {input_file}', printthis=True)
    while i < n:
        start, end, seg_type = segments[i]

        log_message(f'[{i}/{n}] {start}-{end} ({seg_type})')
        if seg_type == 'non_silence':
            # start a speech block
            cur_start = start
            cur_end = end

            # grow forward until we reach MIN or run out
            while (cur_end - cur_start) < MIN_SEGMENT_DURATION and (i + 1) < n:
                next_start, next_end, next_type = segments[i + 1]
                new_duration = next_end - cur_start

                if MAX_SEGMENT_DURATION < new_duration:
                    # if merging next segment would exceed MAX_SEGMENT_DURATION, we break here
                    break
                if(next_type == 'silence' and MIN_SEGMENT_DURATION < new_duration):
                    # if the next segment is silence and merging it would exceed MIN_SEGMENT_DURATION,
                    # we just break here to avoid over-merging
                    break

                # always merge, even if it's silence
                cur_end = next_end
                log_message(
                    f"[INFO] Extending non_silence {cur_start}-{cur_end} with {next_start}-{next_end} ({next_type}) "
                    f"→ now {cur_end - cur_start:.2f}s"
                )
                i += 1  # we consumed that next segment

            # final type MUST be non_silence
            merged_segments.append((cur_start, cur_end, 'non_silence'))
            i += 1

        else:
            # seg_type == 'silence'
            # for silence we just keep it as-is (you *could* add a separate min here later)
            merged_segments.append((start, end, 'silence'))
            i += 1

    # edge case: if the VERY FIRST segment was short, the above puts it in merged_segments anyway.
    # But if you prefer "merge first short into next", we’d need a second pass.
    # For now, this is good enough for most recordings.

    # Extract each segment to its own file.
    extracted_segments = []
    file_index = 0
    for idx, (start, end, seg_type) in enumerate(merged_segments):
        log_message(f'{start}\t{end}\t{seg_type}', printthis=False)
        duration = end - start
        if(duration < 0.05):
            log_message(f"Skipping short segment of file {input_file}:\n\t {start}-{end} ({duration}s)", level="WARNING")
            continue
        if MAX_SEGMENT_DURATION < duration and seg_type == 'non_silence':
            #split this segment into multiple segments
            num_subsegments = int(duration // MAX_SEGMENT_DURATION) + 1
            subsegment_duration = duration / num_subsegments
            for sub_idx in range(num_subsegments):
                sub_start = start + sub_idx * subsegment_duration
                sub_end = start + (sub_idx + 1) * subsegment_duration
                if sub_end > end:
                    sub_end = end
                sub_duration = sub_end - sub_start
                segment_filename = os.path.join(temp_dir, f"{file_index:04d}_{seg_type}_{sub_start:.3f}_{sub_end:.3f}.wav")
                file_index += 1
                if extract_audio_segment(input_file, sub_start, sub_duration, segment_filename, continue_job):
                    extracted_segments.append((segment_filename, seg_type, sub_start, sub_end))
            continue
        segment_filename = os.path.join(temp_dir, f"{file_index:04d}_{seg_type}_{start}_{end}.wav")
        file_index += 1

        if extract_audio_segment(input_file, start, duration, segment_filename, continue_job):
            extracted_segments.append((segment_filename, seg_type, start, end))
    log_message(f'END segments of {input_file}', printthis=True)
    
    return extracted_segments

def process_audio_files(input, output_dir, target_voice_path, continue_job=False):
    supported_extensions = ('.wav', '.mp3')
    venv_python = sys.executable

    # Create a temporary directory to hold segments.
    temp_dir = os.path.join(output_dir, "temp_segments")
    os.makedirs(temp_dir, exist_ok=True)

    half_true = True  # Use half precision for faster processing if supported.

    # Automatically detect the best available device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    load_model_begin_time = datetime.datetime.now()
    print(f"{load_model_begin_time} Using device: {device}")
    model = ChatterboxVC.from_pretrained(device)
    load_model_end_time = datetime.datetime.now()
    print(f"{load_model_end_time} Model loaded, duration: {load_model_end_time - load_model_begin_time}")

    output_files = []

    # if input is actually a file, process that single file
    if os.path.isfile(input):
        filename = os.path.basename(input)
        input_dir = os.path.dirname(input)
        out_file = process_audio_file(filename, input_dir, temp_dir, output_dir, target_voice_path, model, continue_job)
        if out_file:
            output_files.append(out_file)
        return output_files

    # Process each file in the input directory.
    for filename in os.listdir(input):
        if filename.lower().endswith(supported_extensions):
            out_file = process_audio_file(filename, input, temp_dir, output_dir, target_voice_path, model, continue_job)
            if out_file:
                output_files.append(out_file)
    return output_files

def process_audio_file(filename, input_dir, temp_dir, output_dir, target_voice_path, model, continue_job):
    begin_time = datetime.datetime.now()
    print(f"{begin_time} Processing file: {filename}, using target voice: {target_voice_path}")

    input_file = os.path.join(input_dir, filename)
    base_name, _ = os.path.splitext(filename)

    # Create a subdirectory for segments from this file.
    file_temp_dir = os.path.join(temp_dir, base_name)
    os.makedirs(file_temp_dir, exist_ok=True)
    # remove everything under file_temp_dir, but leave the directory itself
    shutil.rmtree(file_temp_dir)
    os.makedirs(file_temp_dir, exist_ok=True)

    # Split the file into silence and non-silence segments.
    segments = split_audio_into_segments(
        input_file, file_temp_dir, noise_threshold=-30, silence_duration=1, continue_job=continue_job, padding_sec=0.2
    )

    # Process each segment accordingly.
    # Batch process all non-silence segments at once using the new CLI
    env_python = sys.executable
    processed_dir = os.path.join(file_temp_dir, "processed_segments")
    os.makedirs(processed_dir, exist_ok=True)
    # Move all silence-only segments directly into processed_dir
    for fname in os.listdir(file_temp_dir):
        if '_silence' in fname and '_non_silence' not in fname:
            src = os.path.join(file_temp_dir, fname)
            dst = os.path.join(processed_dir, fname)
            shutil.move(src, dst)

    padded_dir = os.path.join(file_temp_dir, "padded_segments")
    os.makedirs(padded_dir, exist_ok=True)

    end_time = datetime.datetime.now()
    print(f"{end_time} Finished splitting segments for file: {filename}, duration: {end_time - begin_time}")

    # model processing
    wav_files = [file for file in os.listdir(file_temp_dir) if file.endswith(".wav")]
    total = len(wav_files)
    for idx, file in enumerate(wav_files, 1):
        processing_begin_time = datetime.datetime.now()
        print(f"{processing_begin_time} Processing {idx}/{total}: {file}")

        input_path = os.path.join(file_temp_dir, file)
        duration, sample_rate, channel_str = get_audio_detail(input_path)

        print(f"audio input path: {input_path}, sample_rate: {sample_rate}, channel: {channel_str}, duration: {duration}")

        if sample_rate <= 0:
            sample_rate = model.sr
        if channel_str == 'unknown':
            channel_str = 'mono'

        print(f"append_silence_then_head sample reate: {sample_rate}")

        padding_silient_seconds = PADDING_SECONDS - PADDING_HEADING_SECONDS

        padded_input_path = os.path.join(padded_dir, file)
        padded_input_path = append_silence_then_head(
            input_file=input_path,
            silence_duration=padding_silient_seconds,
            head_duration=PADDING_HEADING_SECONDS,
            sample_rate=sample_rate,
            channel=channel_str,
            continue_job=continue_job,
            output_file=padded_input_path,
        )

        print(f"Padded input path: {padded_input_path}")
        print(f"target voice path: {target_voice_path}")

        wav = model.generate(
            audio=padded_input_path,
            target_voice_path=target_voice_path,
        )

        trim_samples = int(model.sr * 5.0)
        if trim_samples > 0 and wav.shape[-1] > trim_samples:
            wav = wav[..., :-trim_samples]
        output_path = os.path.join(processed_dir, file)

        print(f"Saving output to: {output_path}")

        ta.save(output_path, wav, model.sr)
        processing_end_time = datetime.datetime.now()
        print(f"{processing_end_time} Finished {idx}/{total}: {file}, duration: {processing_end_time - processing_begin_time}")

    combined_begin_time = datetime.datetime.now()
    print(f"{combined_begin_time} Combining ...")
    # Fix double .wav extension from infer_cli_dir outputs
    for fname in os.listdir(processed_dir):
        if fname.endswith('.wav.wav'):
            src = os.path.join(processed_dir, fname)
            dst = os.path.join(processed_dir, fname[:-4])
            shutil.move(src, dst)

    # Build final segment list, copying silence segments as-is
    final_segments = []
    for seg_file, seg_type, start, end in segments:
        basename = os.path.basename(seg_file)
        processed_seg = os.path.join(processed_dir, basename)
        if seg_type == 'silence':
            continue
        final_segments.append((processed_seg, start))

    # Reorder segments by their original start times.
    final_segments_sorted = sorted(final_segments, key=lambda x: x[1])

    # Write a concat file for ffmpeg.
    concat_list = os.path.join(file_temp_dir, "concat.txt")
    with open(concat_list, 'w', encoding='utf-8') as f:
        for seg, _ in final_segments_sorted:
            f.write(f"file '{seg}'\n")

    # get target_voice_path file name without path and extension
    target_voice_file_name = os.path.splitext(os.path.basename(target_voice_path))[0]
    final_output_path = os.path.join(output_dir, f"chatterbox-target-{target_voice_file_name}-_{base_name}.wav")

    all_segments = [seg for seg, _ in final_segments_sorted]

    batch_concat_python(all_segments, final_output=final_output_path)

    print(f"Cleaning up temporary files...")
    # cleanup temp_dir
    #shutil.rmtree(file_temp_dir, ignore_errors=True)

    combined_end_time = datetime.datetime.now()
    print(f"{combined_end_time} Finished combining for file: {filename}, duration: {combined_end_time - combined_begin_time}")

    return final_output_path

def batch_concat_python(files, final_output):
    """
    Concatenate and mix segments precisely using pydub AudioSegment.
    Files should be named with start times encoded (e.g., _start_end.wav).
    """
    # Regex to extract start time (seconds)
    re_time = re.compile(r'_(\d+\.?\d*)_\d+\.?\d*\.wav$')
    segments = []  # (start_ms, AudioSegment)

    # Load segments and schedule overlays
    max_end = 0
    for f in files:
        m = re_time.search(os.path.basename(f))
        start_sec = float(m.group(1)) if m else 0.0
        audio = AudioSegment.from_file(f)
        start_ms = int(start_sec * 1000)
        end_ms = start_ms + len(audio)
        if end_ms > max_end:
            max_end = end_ms
        segments.append((start_ms, audio))

    # Create silent base track of required length
    mixed = AudioSegment.silent(duration=max_end)
    # Overlay each segment at correct position
    for start_ms, audio in segments:
        mixed = mixed.overlay(audio, position=start_ms)

    # Export mixed result
    mixed.export(final_output, format='wav')
    safe_print(f"Created {final_output}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process audio files by splitting into silence and non-silence segments.')
    parser.add_argument('--input', required=True, help="Input directory")
    parser.add_argument('--output_dir', required=True, help="Output directory")
    parser.add_argument('--target', help="Path to the target sound file")
    parser.add_argument('--log_file', default=LOG_FILE, help="Log")
    args = parser.parse_args()

    log_file = args.log_file

    # Record start time
    start_time = datetime.datetime.now()
    log_message(f"V 20260108 (long padding)")
    log_message(f"Processing started at {start_time}")
    safe_print(f"Processing started at {start_time}")

    process_audio_files(args.input, args.output_dir, args.target, False)

    # Record end time
    end_time = datetime.datetime.now()
    log_message(f"Processing ended at {end_time}, duration: {end_time - start_time}")
    safe_print(f"Processing ended at {end_time}, duration: {end_time - start_time}")
