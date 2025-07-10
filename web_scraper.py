# web_scraper.py
import requests
from bs4 import BeautifulSoup
import os

def scrape_and_save_text(url, output_path="data/langchain.txt"):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Clean up text: remove scripts and styles
    for tag in soup(["script", "style"]):
        tag.decompose()
        
    text = soup.get_text(separator="\n")
    
    # Optional cleanup: remove empty lines
    cleaned_text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)
