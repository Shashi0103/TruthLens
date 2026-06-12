import requests
from bs4 import BeautifulSoup
import urllib.parse

# Check if newspaper is available
HAS_NEWSPAPER = False
try:
    from newspaper import Article
    HAS_NEWSPAPER = True
except ImportError:
    pass

# Custom headers to mimic a browser request and avoid scraping blocks
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

def scrape_with_newspaper(url: str):
    """Attempts to scrape news article headline and body using newspaper3k."""
    if not HAS_NEWSPAPER:
        raise ImportError("newspaper3k is not installed.")
        
    article = Article(url, headers=HEADERS)
    article.download()
    article.parse()
    
    title = article.title
    text = article.text
    
    if not title and not text:
        raise ValueError("newspaper3k returned empty title and content.")
        
    return title, text

def scrape_with_bs4(url: str):
    """Fallback scraping method using requests and BeautifulSoup."""
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 1. Try to extract the title/headline
    # First check standard h1 tags
    h1_tag = soup.find('h1')
    if h1_tag:
        title = h1_tag.get_text().strip()
    else:
        # Fall back to meta/title tag
        title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else "No Headline Found"
        
    # Remove junk suffixes like "| CNN" or "- BBC News"
    for separator in [' - ', ' | ', ' : ']:
        if separator in title:
            title = title.split(separator)[0].strip()

    # 2. Try to extract the body text from paragraphs
    # Filter out empty or extremely short paragraphs
    paragraphs = []
    
    # Look for main content container if possible
    main_content = None
    for selector in ['article', '[role="main"]', '.article-body', '.story-body', '.entry-content', '#main-content']:
        main_content = soup.select_one(selector)
        if main_content:
            break
            
    container = main_content if main_content else soup
    
    for p in container.find_all('p'):
        text_content = p.get_text().strip()
        # Avoid short social media captions, cookies text, etc.
        if len(text_content) > 30:
            paragraphs.append(text_content)
            
    text = "\n\n".join(paragraphs)
    
    if not text:
        # If no paragraphs found in article/container, get all text elements from body
        text = soup.body.get_text(separator="\n\n").strip() if soup.body else soup.get_text().strip()
        
    return title, text

def scrape_news_url(url: str):
    """
    Scrapes a news article URL.
    Tries newspaper3k first, then falls back to BeautifulSoup.
    """
    # Basic URL validation
    parsed = urllib.parse.urlparse(url)
    if not all([parsed.scheme, parsed.netloc]):
        raise ValueError("Invalid URL format. Please include http:// or https://")
        
    try:
        if HAS_NEWSPAPER:
            return scrape_with_newspaper(url)
        else:
            return scrape_with_bs4(url)
    except Exception as e:
        # Fallback to BeautifulSoup if newspaper fails
        try:
            return scrape_with_bs4(url)
        except Exception as fallback_e:
            raise RuntimeError(f"Failed to scrape URL with newspaper3k ({e}) and BeautifulSoup ({fallback_e})")

if __name__ == "__main__":
    # Test URL
    test_url = "https://example.com"
    try:
        headline, content = scrape_news_url(test_url)
        print("Headline:", headline)
        print("Content Preview:", content[:200], "...")
    except Exception as err:
        print("Scraping Test Error:", err)
