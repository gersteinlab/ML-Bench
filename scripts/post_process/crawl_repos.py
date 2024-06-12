import os
import subprocess
import sys


def checkout_commit(clone_dir, commit):
    try:
        subprocess.run(['git', 'checkout', commit], cwd=clone_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error checking out commit {commit}: {e}")


def read_files_in_dir(directory):
    all_files_content = ""
    readme_files_content = ""
    other_files_content = ""

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'README' in file.upper():
                        readme_files_content += f"\n=== {file_path} ===\n{content}\n"
                    else:
                        other_files_content += f"\n=== {file_path} ===\n{content}\n"
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    all_files_content = readme_files_content + other_files_content
    return all_files_content


def main(clone_dir, output_file, commit):
    commits = [commit]
    with open(output_file, 'w', encoding='utf-8') as f:
        for commit in commits:
            content = read_files_in_dir(clone_dir)
            f.write(f"{content}")
            print(f"Processed commit {commit}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <clone_dir> <output_file> <commit>")
        sys.exit(1)

    clone_dir = sys.argv[1]
    output_file = sys.argv[2]
    commit = sys.argv[3]

    main(clone_dir, output_file, commit)
