import os


def main():
    for circle_name in os.listdir('black_circles'):
        if circle_name.startswith('base'):
            continue
        seq_name = circle_name.rsplit('.', 1)[0]
        dir_name = os.path.join('circle_tracking', seq_name)
        os.system(f'python play_video.py --write --dir {dir_name} --target_seq {seq_name}')


if __name__ == '__main__':
    main()
