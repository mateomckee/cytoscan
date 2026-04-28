import sys
from pathlib import Path


def organize_frames(directory: str):
    src = Path(directory)
    files = sorted(
        [p for p in src.iterdir() if p.suffix.lower() in ('.tif', '.tiff')]
    )

    if len(files) % 3 != 0:
        print(f'warning: {len(files)} files is not a multiple of 3')

    suffixes = ['fl', 'br', 'mx']
    dirs = {
        'fl': src / 'fluorescent',
        'br': src / 'brightfield',
        'mx': src / 'mixed',
    }
    for d in dirs.values():
        d.mkdir(exist_ok=True)

    for i, f in enumerate(files):
        frame = i // 3
        suffix = suffixes[i % 3]
        new_name = f'frame{frame:03d}_{suffix}{f.suffix.lower()}'
        f.rename(dirs[suffix] / new_name)
        print(f'{f.name} -> {suffix}/{new_name}')


if __name__ == '__main__':
    organize_frames(sys.argv[1] if len(sys.argv) > 1 else '.')
