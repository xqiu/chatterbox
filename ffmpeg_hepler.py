def padding_tail_command(inputfilepath: str, input_sample_rate: int, input_channel: str, append_duration_secounds: float, output_filepath: str) -> list[str]:
    """
    return command to RUN
    Pad silence to the end of audio file

    Args:
        inputfilepath (str): input audio file path
        input_sample_rate (int): input audio sample rate
        input_channel (str): input audio channel layout
        append_duration_secounds (float): duration to append in seconds
        output_filepath (str): output audio file path

    Returns:
        str: output audio file path
    """
    input_file = inputfilepath
    output_file = output_filepath

    ffmpeg_pad_cmd = [
        'ffmpeg', '-y', '-i', input_file,
        #BEGIN generate silience audio
        '-f', 'lavfi',
        '-t', str(append_duration_secounds),
        '-i', f'anullsrc=channel_layout={input_channel}:sample_rate={input_sample_rate}',
        #END generate silience audio
        '-filter_complex', '[0:a][1:a]concat=n=2:v=0:a=1',
        #'-c:a', 'aac',
        output_file
    ]

    return ffmpeg_pad_cmd