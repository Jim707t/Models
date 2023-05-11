import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probability_distribution = {}
    
    # all the pages that is linked to `page`
    linked_pages = corpus[page]
    
    for p in corpus:
        if p in linked_pages:
            probability_distribution[p] = damping_factor / len(linked_pages)
        else:
            # linked_pages = set(corpus.keys())
            probability_distribution[p] = (1 - damping_factor) / len(corpus)  
    
    return probability_distribution         


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages_rank = dict.fromkeys(corpus.keys(), 0)
    # choice randomly the first page
    page = random.choice(list(corpus.keys()))
    
    for _ in range(n):
        # take the probability distribution for each page and pass as weights 
        # to `random.choice` function
        probability_distribution = transition_model(corpus, page, damping_factor)
        page = random.choices(list(probability_distribution.keys()),
                             weights=list(probability_distribution.values())
                             )[0]
        pages_rank[page] += 1
    
    # get page rank by divide total sample for a page to the sum of all page rank
    total_sample = sum(pages_rank.values())
    for page in pages_rank:
        pages_rank[page] /= total_sample
    return pages_rank    
        
        
def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    threshold = 0.001
    N = len(corpus)
    pages_rank = dict.fromkeys(corpus.keys(), 1/N)
    
    while True:
        new_pages_rank = {}
        diff = 0
        for page in corpus:
            pr_page = (1 - damping_factor) / N
            for i in corpus:
                # If the current page is linked to from the current iteration page, add its 
                # contribution to the PageRank value.
                if page in corpus[i]:
                    pr_page += damping_factor * pages_rank[i] / len(corpus[i])
            new_pages_rank[page] = pr_page
            # Calculate the difference between the new and old PageRank 
            # values for the current page 
            diff += abs(new_pages_rank[page] - pages_rank[page])
        pages_rank = new_pages_rank
        if diff < threshold:
            break
    
    # Normalize the PageRank values so they sum to 1
    total = sum(pages_rank.values())
    pages_rank = {k: v / total for k, v in pages_rank.items()}
    
    return pages_rank


if __name__ == "__main__":
    main()
