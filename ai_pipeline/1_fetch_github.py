import os
import requests
from typing import List, Dict, Any

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"

def fetch_github_advisories(token: str, limit: int = 1500) -> List[Dict[str, Any]]:
    """
    Fetch GitHub Security Advisories using the GraphQL API.
    We filter for advisories that have a reference pointing to a commit fix.
    """
    headers = {"Authorization": f"Bearer {token}"}

    query = """
    query($cursor: String) {
      securityAdvisories(first: 100, after: $cursor) {
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          ghsaId
          cwes(first: 10) {
            nodes {
              cweId
            }
          }
          description
          summary
          references {
            url
          }
        }
      }
    }
    """

    advisories = []
    has_next_page = True
    cursor = None

    print(f"  > Pinging GitHub GraphQL API...")

    while has_next_page and len(advisories) < limit:
        variables = {"cursor": cursor}
        response = requests.post(
            GITHUB_GRAPHQL_URL,
            json={"query": query, "variables": variables},
            headers=headers
        )

        if response.status_code != 200:
            print(f"Failed to fetch from GitHub API: {response.text}")
            break

        data = response.json()
        if "errors" in data:
            print(f"GraphQL Errors: {data['errors']}")
            break

        advisories_data = data["data"]["securityAdvisories"]
        has_next_page = advisories_data["pageInfo"]["hasNextPage"]
        cursor = advisories_data["pageInfo"]["endCursor"]

        for node in advisories_data["nodes"]:
            # Check if there's a commit reference
            has_commit_fix = False
            for ref in node.get("references", []):
                url = ref.get("url", "")
                if "github.com" in url and "/commit/" in url:
                    has_commit_fix = True
                    break

            if has_commit_fix and node.get("description"):
                cwe_ids = [cwe["cweId"] for cwe in node.get("cwes", {}).get("nodes", [])]
                advisories.append({
                    "ghsa_id": node["ghsaId"],
                    "summary": node["summary"],
                    "description": node.get("description", ""),
                    "cwes": cwe_ids
                })

    return advisories

if __name__ == "__main__":
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("Please set GITHUB_TOKEN")
    else:
        advs = fetch_github_advisories(token, limit=100)
        print(f"Test fetch complete. Grabbed {len(advs)} advisories with commits.")
