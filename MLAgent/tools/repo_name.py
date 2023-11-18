import os

def get_repo_name(repo_url):
    repo_name = os.path.basename(repo_url.rstrip('/'))
    return repo_name