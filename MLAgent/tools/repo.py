from tools import keywords
import requests

def search_github_repositories_by_keywords(keywords):
    query_string = '+'.join(keywords)

    print(query_string)
    api_url = f"https://api.github.com/search/repositories?q={query_string}&page=1&per_page=10"

    response = requests.get(api_url)

    if response.status_code == 200:
        search_results = response.json()
        
        repo_urls = [repo["html_url"] for repo in search_results["items"]]
        return repo_urls
    else:
        return []

def get_repo_urls(query,model_name,api_type,function_type):
    keywds = keywords.get_keywords(query,model_name,api_type,function_type)
    print(keywds)
    repo_urls = search_github_repositories_by_keywords(keywds)
    return repo_urls
    
