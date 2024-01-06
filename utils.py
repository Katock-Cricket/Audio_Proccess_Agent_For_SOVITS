import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import librosa
import soundfile as sf
from pydub import AudioSegment, silence
from tqdm import tqdm


def cut_silence_fn(file_param):
    input_file, silence_thresh, min_silence_len, padding = file_param
    clean_name = Path(input_file).stem

    sound = AudioSegment.from_file(input_file)
    sound = sound.set_sample_width(2)
    nonsilent_chunks = silence.detect_nonsilent(
        sound,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )
    chunks = [sound[max(start - padding, 0):min(end + padding, len(sound))] for start, end in nonsilent_chunks]
    processed_sound = sum(chunks)
    processed_sound.export(os.path.join(os.path.dirname(input_file), f"{clean_name}_cut.wav"), format="wav")


def cut_silence(input_file_list, silence_thresh=-50, min_silence_len=300, padding=100, cpu_count=1):
    file_param_list = [(file, silence_thresh, min_silence_len, padding) for file in input_file_list]

    with Pool(processes=cpu_count) as pool:
        for _ in tqdm(pool.imap_unordered(cut_silence_fn, file_param_list), total=len(file_param_list), desc="截静音"):
            pass

    for file in input_file_list:
        os.remove(file)


def split_segment(params):
    input_file, output_file, start_time, end_time, sr = params
    duration = end_time - start_time
    if duration < 1:
        return
    audio, _ = librosa.load(input_file, sr=sr, mono=False, offset=start_time, duration=duration)
    audio = audio.transpose()
    sf.write(output_file, audio, sr)


def split_file(info):
    input_file, sec = info
    clean_name = Path(input_file).stem

    audio, sr = librosa.load(input_file, sr=None, mono=False)
    total_frames = librosa.get_duration(y=audio, sr=sr)
    num_segments = int(total_frames // sec) + 1

    segment_params = []
    for segment in range(num_segments):
        start_time = segment * sec
        end_time = min((segment + 1) * sec, total_frames)
        output_file = os.path.join(os.path.dirname(input_file), f"{clean_name}_{segment}.wav")
        segment_params.append((input_file, output_file, start_time, end_time, sr))

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(split_segment, segment_params), total=len(segment_params), desc=f"切片{clean_name}",
                  leave=False))

    print(f"切片音频{input_file}完成")


def split_audio(input_file_list, sec=3, cpu_count=1):
    file_param_list = [(file, sec) for file in input_file_list]

    with Pool(processes=cpu_count) as pool:
        for _ in tqdm(pool.imap(split_file, file_param_list), total=len(file_param_list), desc="切片总体进度"):
            pass

    for file in input_file_list:
        os.remove(file)


def normalize(input_file_list, target_dbfs):
    for file in input_file_list:
        audio = AudioSegment.from_file(file, "wav")
        change_in_dbfs = target_dbfs - audio.dBFS
        normalized_audio = audio.apply_gain(change_in_dbfs)
        normalized_audio.export(file, format="wav")


def rename(input_file_list, name):
    path = os.path.dirname(input_file_list[0])
    for i, source in enumerate(input_file_list):
        new_filename = f"{name}{i}.wav"
        destination = os.path.join(path, new_filename)
        os.rename(source, destination)
