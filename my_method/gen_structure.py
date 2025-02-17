import os

def generate_tree(path, ignore_dirs=[], max_depth=3, prefix=''):
    if max_depth < 0:
        return ''
    dirs, files = [], []
    for item in os.listdir(path):
        if item in ignore_dirs:
            continue
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path):
            dirs.append(item)
        else:
            files.append(item)
    output = []
    for i, dir_name in enumerate(dirs):
        is_last = i == len(dirs) - 1 and not files
        line_prefix = prefix + ('└── ' if is_last else '├── ')
        output.append(line_prefix + dir_name + '/')
        new_prefix = prefix + ('    ' if is_last else '│   ')
        output.append(generate_tree(os.path.join(path, dir_name), ignore_dirs, max_depth-1, new_prefix))
    for i, file_name in enumerate(files):
        is_last = i == len(files) - 1
        line_prefix = prefix + ('└── ' if is_last else '├── ')
        output.append(line_prefix + file_name)
    return '\n'.join(output)

ignore = ['node_modules', '.git', 'venv']
tree_str = generate_tree('.', ignore_dirs=ignore, max_depth=3)
with open('DIRECTORY_TREE.md', 'w') as f:
    f.write(f'```\n{tree_str}\n```\n')