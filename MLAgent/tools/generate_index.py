import os

def write_indexfile(repo_name,root_directory):
    directory_path = root_directory
    all_contents = [os.path.relpath(os.path.join(root, item), directory_path) for root, _, items in os.walk(directory_path) for item in items]
    output_file = 'directory_contents.txt'
    with open(repo_name+"_index.txt", 'w') as file:
        file.write('\n'.join(all_contents) + '\n')
    print(f'All directory contents written')

