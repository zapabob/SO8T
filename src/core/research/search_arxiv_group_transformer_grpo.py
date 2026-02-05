#!/usr/bin/env python3
"""
Search Arxiv for Transformer Group Representation Models & Advanced GRPO (2025-2026)
Including GitHub repository analysis
"""

import requests
import json
import time
from datetime import datetime
import re

def search_arxiv_papers(query, max_results=20):
    """Search Arxiv for recent papers"""
    base_url = 'https://export.arxiv.org/api/query'
    params = {
        'search_query': query,
        'start': 0,
        'max_results': max_results,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }

    try:
        response = requests.get(base_url, params=params, timeout=15)
        if response.status_code == 200:
            content = response.text
            papers = []

            # Parse XML response
            entries = content.split('<entry>')[1:]

            for entry in entries[:max_results]:
                try:
                    # Extract title
                    title_match = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                    title = title_match.group(1).strip() if title_match else 'Unknown'

                    # Extract authors
                    authors = []
                    author_matches = re.findall(r'<name>(.*?)</name>', entry)
                    authors = [author.strip() for author in author_matches[:3]]

                    # Extract summary
                    summary_match = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                    summary = summary_match.group(1).strip() if summary_match else ''

                    # Extract link
                    link_match = re.search(r'<id>(.*?)</id>', entry)
                    link = link_match.group(1).strip() if link_match else ''

                    # Extract published date
                    published_match = re.search(r'<published>(.*?)</published>', entry)
                    published = published_match.group(1).strip() if published_match else ''

                    # Extract categories
                    categories = []
                    category_matches = re.findall(r'<category term="([^"]*)"', entry)
                    categories = category_matches[:3]

                    papers.append({
                        'title': title,
                        'authors': authors,
                        'summary': summary[:300] + '...' if len(summary) > 300 else summary,
                        'link': link,
                        'published': published,
                        'categories': categories
                    })

                except Exception as e:
                    continue

            return papers
        else:
            print(f'Arxiv API error: {response.status_code}')
            return []

    except Exception as e:
        print(f'Arxiv search error: {e}')
        return []

def find_github_links(text):
    """Extract GitHub repository links from text"""
    github_pattern = r'https?://github\.com/[^\s)]+'
    matches = re.findall(github_pattern, text)
    return list(set(matches))

def analyze_paper_relevance(paper):
    """Analyze paper relevance for SO8T"""
    text = paper['title'].lower() + ' ' + paper['summary'].lower()

    # Group representation keywords
    group_keywords = [
        'group representation', 'equivariant', 'invariant', 'symmetric',
        'lie group', 'lie algebra', 'so(', 'orthogonal group', 'unitary group',
        'representation theory', 'geometric', 'manifold', 'grassmann'
    ]

    # GRPO keywords
    grpo_keywords = [
        'reinforcement learning', 'policy optimization', 'geometric rl',
        'grpo', 'ppo', 'actor-critic', 'reward', 'value function',
        'reinforce', 'advantage', 'baseline'
    ]

    # Transformer keywords
    transformer_keywords = [
        'transformer', 'attention', 'self-attention', 'multi-head',
        'positional encoding', 'encoder', 'decoder', 'cross-attention'
    ]

    group_score = sum(1 for kw in group_keywords if kw in text)
    grpo_score = sum(1 for kw in grpo_keywords if kw in text)
    transformer_score = sum(1 for kw in transformer_keywords if kw in text)

    total_score = group_score + grpo_score + transformer_score

    return {
        'group_score': group_score,
        'grpo_score': grpo_score,
        'transformer_score': transformer_score,
        'total_score': total_score,
        'is_relevant': total_score >= 2
    }

def check_github_repo(repo_url):
    """Check if GitHub repo exists and get basic info"""
    try:
        # Remove trailing characters
        clean_url = repo_url.rstrip('.,)')
        api_url = clean_url.replace('https://github.com/', 'https://api.github.com/repos/')

        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                'name': data.get('name', ''),
                'description': data.get('description', ''),
                'stars': data.get('stargazers_count', 0),
                'language': data.get('language', ''),
                'updated': data.get('updated_at', ''),
                'topics': data.get('topics', [])[:5]
            }
        else:
            return {'error': f'HTTP {response.status_code}'}
    except Exception as e:
        return {'error': str(e)}

def main():
    print('SEARCHING: Transformer Group Representation Models & Advanced GRPO')
    print('=' * 90)
    print('Arxiv 2025-2026 papers + GitHub repositories analysis')
    print()

    # Search queries
    search_queries = [
        'group representation transformer',
        'equivariant transformer neural network',
        'geometric transformer',
        'symmetric transformer',
        'lie group transformer',
        'GRPO reinforcement learning',
        'geometric reinforcement learning',
        'group theoretic reinforcement learning',
        'advanced GRPO',
        'geometric policy optimization'
    ]

    all_papers = []
    processed_titles = set()

    print('COLLECTING PAPERS FROM ARXIV...')
    print()

    for query in search_queries:
        print(f'Searching: "{query}"')

        papers = search_arxiv_papers(query, max_results=8)

        for paper in papers:
            if paper['title'] not in processed_titles:
                relevance = analyze_paper_relevance(paper)
                if relevance['is_relevant']:
                    paper['relevance'] = relevance
                    paper['github_links'] = find_github_links(paper['summary'])
                    all_papers.append(paper)
                    processed_titles.add(paper['title'])

        time.sleep(1)  # Rate limiting

    # Sort by relevance and date
    all_papers.sort(key=lambda x: (x['relevance']['total_score'], x['published']), reverse=True)

    print(f'\nFOUND {len(all_papers)} HIGHLY RELEVANT PAPERS')
    print('=' * 90)

    for i, paper in enumerate(all_papers[:12], 1):
        print(f'\n{i}. {paper["title"]}')
        print(f'   Authors: {", ".join(paper["authors"][:3])}')
        print(f'   Published: {paper["published"][:10]}')
        print(f'   Categories: {", ".join(paper["categories"][:3])}')
        print(f'   Relevance Score: {paper["relevance"]["total_score"]}/9')
        print(f'   Group: {paper["relevance"]["group_score"]}, GRPO: {paper["relevance"]["grpo_score"]}, Transformer: {paper["relevance"]["transformer_score"]}')
        print(f'   Arxiv: {paper["link"]}')

        if paper['github_links']:
            print(f'   GitHub Repos: {len(paper["github_links"])}')
            for link in paper['github_links'][:2]:
                print(f'     Link: {link}')

        print(f'   Summary: {paper["summary"]}')

    # Analyze GitHub repositories
    print(f'\n\nGITHUB REPOSITORY ANALYSIS:')
    print('=' * 50)

    github_papers = [p for p in all_papers if p['github_links']]
    analyzed_repos = []

    for paper in github_papers[:8]:
        print(f'\nPaper: {paper["title"][:60]}...')

        for repo_url in paper['github_links'][:2]:
            print(f'   Analyzing: {repo_url}')
            repo_info = check_github_repo(repo_url)

            if 'error' not in repo_info:
                print(f'      Stars: {repo_info["stars"]}')
                print(f'      Language: {repo_info["language"]}')
                print(f'      Description: {repo_info["description"][:100]}...' if repo_info["description"] else '      No description')
                if repo_info["topics"]:
                    print(f'      Topics: {", ".join(repo_info["topics"])}')
            else:
                print(f'      Error: {repo_info["error"]}')

            analyzed_repos.append({
                'paper_title': paper['title'],
                'repo_url': repo_url,
                'repo_info': repo_info
            })

    # Save comprehensive results
    results = {
        'search_timestamp': datetime.now().isoformat(),
        'total_papers_found': len(all_papers),
        'papers_with_github': len(github_papers),
        'analyzed_repositories': len(analyzed_repos),
        'papers': all_papers,
        'github_analysis': analyzed_repos,
        'so8t_relevance_summary': {
            'group_representation_focus': len([p for p in all_papers if p['relevance']['group_score'] >= 2]),
            'grpo_focus': len([p for p in all_papers if p['relevance']['grpo_score'] >= 2]),
            'transformer_focus': len([p for p in all_papers if p['relevance']['transformer_score'] >= 2])
        }
    }

    with open('arxiv_group_transformer_grpo_2025_2026_complete.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f'\nResults saved to: arxiv_group_transformer_grpo_2025_2026_complete.json')
    print(f'\nSUMMARY:')
    print(f'   Total relevant papers: {len(all_papers)}')
    print(f'   Papers with GitHub repos: {len(github_papers)}')
    print(f'   Analyzed repositories: {len(analyzed_repos)}')
    print(f'   Group representation focus: {results["so8t_relevance_summary"]["group_representation_focus"]}')
    print(f'   GRPO focus: {results["so8t_relevance_summary"]["grpo_focus"]}')
    print(f'   Transformer focus: {results["so8t_relevance_summary"]["transformer_focus"]}')

if __name__ == '__main__':
    main()