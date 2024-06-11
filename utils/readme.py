import os

def find_readme_files(directory):
    readme_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower() == "readme.md":
                readme_files.append(os.path.join(root, file))
    readme_files =  sorted(readme_files, key=len)
    return readme_files 
