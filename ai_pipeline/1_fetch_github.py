import os
import json
import requests
from typing import List, Dict, Any

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"

def fetch_github_advisories(token: str, limit: int = 1500) -> List[Dict[str, Any]]:
    """
    Fetches Security Advisories from GitHub using the GraphQL API.
    Filters for nodes that contain a reference to a specific commit fix.
    """
    headers = {"Authorization": f"Bearer {token}"}
    query = """
    query($cursor: String) {
      securityAdvisories(first: 100, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          ghsaId
          cwes(first: 10) { nodes { cweId } }
          description
          summary
          references { url }
        }
      }
    }
    """
    advisories = []
    cursor = None
    has_next_page = True

    print(f" > Fetching GitHub Security Advisories...")
    while has_next_page and len(advisories) < limit:
        response = requests.post(GITHUB_GRAPHQL_URL, json={"query": query, "variables": {"cursor": cursor}}, headers=headers)

        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            break

        data = response.json()["data"]["securityAdvisories"]
        cursor = data["pageInfo"]["endCursor"]
        has_next_page = data["pageInfo"]["hasNextPage"]

        for node in data["nodes"]:
            # Check for commit references to ensure we have a fix/patch associated
            if any("github.com" in r["url"] and "/commit/" in r["url"] for r in node["references"]):
                advisories.append({
                    "ghsa_id": node["ghsaId"],
                    "summary": node["summary"],
                    "description": node["description"],
                    "cwes": [c["cweId"] for c in node["cwes"]["nodes"]]
                })
    return advisories

if __name__ == "__main__":
    token = os.getenv("GITHUB_TOKEN")

    # Get the directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    if token:
        data = fetch_github_advisories(token)
        os.makedirs(data_dir, exist_ok=True)

        output_file = os.path.join(data_dir, "raw_github.json")
        with open(output_file, "w") as f:
            json.dump(data, f)
        print(f"Saved {len(data)} raw GitHub advisories in {output_file}")
    else:
        print("Error: GITHUB_TOKEN not found in environment variables.")
