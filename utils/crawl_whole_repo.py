import os
import subprocess
import sys
import shutil

def clone_repo(repo_url, clone_dir):
    """
    Clone the git repository.
    """
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir)
    subprocess.run(['git', 'clone', repo_url, clone_dir], check=True)


def checkout_commit(clone_dir, commit):
    """
    Checkout a specific commit.
    """
    try:
        subprocess.run(['git', 'checkout', commit], cwd=clone_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error checking out commit {commit}: {e}")


def read_files_in_dir(directory):
    """
    Read all files in the directory, putting README files first.
    """
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


def main(repo_url, clone_dir, output_file, commit):
    clone_repo(repo_url, clone_dir)
    commits = [commit]
    with open(output_file, 'w', encoding='utf-8') as f:
        for commit in commits:
            checkout_commit(clone_dir, commit)
            content = read_files_in_dir(clone_dir)
            f.write(f"{content}")
            print(f"Processed commit {commit}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py <repo_url> <clone_dir> <output_file> <commit>")
        sys.exit(1)

    repo_url = sys.argv[1]
    clone_dir = sys.argv[2]
    output_file = sys.argv[3]
    commit = sys.argv[4]

    main(repo_url, clone_dir, output_file, commit)
