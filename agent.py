import os.path
from multiprocessing import cpu_count
import shutil

from utils import *


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Audio preprocessor for sovits')
    parser.add_argument('-i', '--input_path', type=str, default='workspace',
                        help='音频位置默认处理全部在用户文件夹里的音频,输出默认覆盖')
    parser.add_argument('-n', '--name', type=str, help='说话人名字，需要指定')
    parser.add_argument('-fi', '--format_input', type=list, default=['wav', 'flac', 'mp3'],
                        help='进行处理的文件的后缀,处理后统一为wav')
    parser.add_argument('-m', '--multi_process', action='store_true', default=False, help='并行处理，建议开，因为速度提升巨大')
    parser.add_argument('-a', '--auto', action='store_true', default=False, help='按顺序自动全流程处理')

    # 第一步 截去静音
    parser.add_argument('-c', '--cut_silence', action='store_true', default=False, help='剪去静音部分')
    parser.add_argument('-thr', '--thresh', type=int, default=-50, help='认为是静音的阈值')
    parser.add_argument('-len', '--min_silence_len', type=int, default=300, help='认为是静音的最小长度默认300毫秒')
    parser.add_argument('-padding', '--padding', type=int, default=100, help='保留在切割点前后的静音长度默认100毫秒')

    # 第二步 音频切片，单个音频过长过短都容易导致训练出错
    parser.add_argument('-s', '--split', action='store_true', default=False, help='音频切片')
    parser.add_argument('-sec', '--split_second', type=float, default=3, help='切片长度默认3秒，建议3-5秒')

    # 第三步 响度归一化，过于不规整的音频则慎用
    parser.add_argument('-norm', '--normalize', action='store_true', default=False, help='进行响度归一化')
    parser.add_argument('-dbfs', '--target_dbfs', type=float, default=-14, help='归一化到响度默认-14uF')

    # 第四步 重命名
    parser.add_argument('-rn', '--rename', action='store_true', default=False, help='批量格式化重命名')

    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.input_path, args.name)):
        os.mkdir(os.path.join(args.input_path, args.name))

    if args.auto:
        auto_process(args)
    else:
        custom_process(args)


# 整理目前的workspace为workspace/<name>/*.wav，即平铺所有文件，并获取所有音频文件的路径
def get_input_file_list(args):
    input_file_list = []
    for root, dirs, files in os.walk(args.input_path):
        for filename in files:
            file = os.path.join(root, filename)
            if file.endswith(tuple(args.format_input)):
                new_path = os.path.join(args.input_path, args.name, os.path.basename(file))
                shutil.move(file, new_path)
            else:
                os.remove(file)

    for root, dirs, files in os.walk(args.input_path):
        for dir_ in dirs:
            if dir_ != args.name:
                shutil.rmtree(os.path.join(root, dir_))

    for file in os.listdir(os.path.join(args.input_path, args.name)):
        input_file_list.append(os.path.join(args.input_path, args.name, file))

    return input_file_list


def auto_process(args):
    print("=====截去静音=====")
    cut_silence(get_input_file_list(args),
                silence_thresh=args.thresh,
                min_silence_len=args.min_silence_len,
                padding=args.padding,
                cpu_count=cpu_count() if args.multi_process else 1)
    print("=====切片=====")
    split_audio(get_input_file_list(args),
                sec=args.split_second,
                cpu_count=cpu_count() if args.multi_process else 1)
    print("=====响度归一化=====")
    normalize(get_input_file_list(args),
              target_dbfs=args.target_dbfs)
    print("=====重命名=====")
    rename(get_input_file_list(args),
           name=args.name)


def custom_process(args):
    if args.cut_silence:
        print("=====截去静音=====")
        cut_silence(get_input_file_list(args),
                    silence_thresh=args.thresh,
                    min_silence_len=args.min_silence_len,
                    padding=args.padding,
                    cpu_count=cpu_count() if args.multi_process else 1)

    if args.split:
        print("=====切片=====")
        split_audio(get_input_file_list(args),
                    sec=args.split_second,
                    cpu_count=cpu_count() if args.multi_process else 1)

    if args.normalize:
        print("=====响度归一化=====")
        normalize(get_input_file_list(args),
                  target_dbfs=args.target_dbfs)

    if args.rename:
        print("=====重命名=====")
        rename(get_input_file_list(args),
               name=args.name)


if __name__ == '__main__':
    main()
