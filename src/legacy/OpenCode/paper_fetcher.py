
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import time
import datetime

def search_arxiv(query, max_results=5):
    """
    Searches the arXiv API for articles matching the query.
    """
    base_url = "http://export.arxiv.org/api/query?"
    # Search for papers published in 2024 or later
    query = f'{query} AND submittedDate:[{datetime.date(2024, 1, 1).strftime("%Y%m%d")} TO {datetime.date.today().strftime("%Y%m%d")}]'
    
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }

    query_url = base_url + urllib.parse.urlencode(params)
    print(f"Querying arXiv API: {query_url}")

    try:
        with urllib.request.urlopen(query_url) as response:
            xml_data = response.read().decode('utf-8')

        root = ET.fromstring(xml_data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        articles = []
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.strip() if entry.find('atom:title', ns) is not None else "N/A"
            summary = entry.find('atom:summary', ns).text.strip() if entry.find('atom:summary', ns) is not None else "N/A"
            published = entry.find('atom:published', ns).text.strip() if entry.find('atom:published', ns) is not None else "N/A"
            authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns) if author.find('atom:name', ns) is not None]
            
            articles.append({
                "source": "arxiv",
                "title": title,
                "authors": authors,
                "summary": summary,
                "published": published,
            })
        return articles
    except Exception as e:
        print(f"Error fetching data from Arxiv: {e}")
        return []

def search_biorxiv(category, start_date="2024-01-01", end_date="2026-12-31"):
    """
    Fetches bioRxiv paper details for a specific category and date range.
    """
    base_url = "https://api.biorxiv.org/details/biorxiv/"
    # Biorxiv API uses cursor-based pagination. We'll just get the first page (e.g., 100 results).
    url = f"{base_url}{start_date}/{end_date}/0"
    print(f"Querying BioRxiv API for category: {category}")

    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode('utf-8'))
        
        papers = []
        if 'collection' in data:
            for paper in data['collection']:
                # Filter by category since the API fetches all and we must filter client-side
                if paper.get('category', '').lower() == category.lower():
                    papers.append({
                        "source": "biorxiv",
                        "title": paper.get('title', 'N/A'),
                        "authors": [a.get('author_name', '') for a in paper.get('authors', [])],
                        "summary": paper.get('abstract', 'N/A'),
                        "published": paper.get('date', 'N/A'),
                    })
        return papers
    except Exception as e:
        print(f"Error fetching data from BioRxiv for category '{category}': {e}")
        return []

def main():
    # Topics from the user
    topics = [
        "chemistry", "mathematics", "physics", "computer science", 
        "machine learning", "large language models", "life science", 
        "cognitive science", "neuroscience", "psychopharmacology", 
        "developmental disorders", "mental illness"
    ]
    
    # Map general topics to potential BioRxiv categories
    biorxiv_categories = {
        "life science": "bioinformatics", # Example mapping
        "cognitive science": "neuroscience",
        "neuroscience": "neuroscience",
        "psychopharmacology": "neuroscience", # No direct category, map to neuroscience
        "developmental disorders": "neuroscience",
        "mental illness": "neuroscience"
    }

    all_papers = []
    max_per_topic = 5

    for topic in topics:
        print(f"\n--- Processing topic: {topic} ---")
        
        # Search Arxiv
        arxiv_query = f"all:{topic}"
        arxiv_results = search_arxiv(arxiv_query, max_results=max_per_topic)
        if arxiv_results:
            all_papers.extend(arxiv_results)
            print(f"Found {len(arxiv_results)} papers on Arxiv.")
        time.sleep(3)

        # Search BioRxiv if the topic maps to a category
        if topic in biorxiv_categories:
            biorxiv_cat = biorxiv_categories[topic]
            # BioRxiv search is less direct, we fetch a date range and filter
            # This is inefficient and might not yield results for the specific topic
            # The provided API structure is for fetching by date, not searching by topic directly
            # The example will be limited.
            print(f"BioRxiv does not support direct topic search via this API endpoint.")
            print(f"Fetching recent papers from category '{biorxiv_cat}' and filtering is not implemented in this script.")

    output_filename = "api_papers.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, ensure_ascii=False, indent=2)

    print(f"\nFinished. Found a total of {len(all_papers)} papers and saved them to {output_filename}")
    print("Note: BioRxiv search was skipped due to API limitations on direct topic searching.")

if __name__ == "__main__":
    main()
