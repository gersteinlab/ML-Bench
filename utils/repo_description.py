import os
import requests

def get_repo_description(repo_url):
    parts = repo_url.strip('/').split('/')
    if len(parts) != 5 or parts[2] != 'github.com':
        return "Invalid GitHub repo URL"
    username, repository_name = parts[3], parts[4]

    api_url = f"https://api.github.com/repos/{username}/{repository_name}"

    response = requests.get(api_url)

    if response.status_code == 200:
        repo_info = response.json()
        repo_description = repo_info["description"]
        return repo_description
    else:
        return "Unable to get warehouse information"