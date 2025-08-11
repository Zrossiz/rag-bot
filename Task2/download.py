import requests
from bs4 import BeautifulSoup
import os

PAGES = [
    "https://starwars.fandom.com/wiki/Darth_Vader",
    "https://starwars.fandom.com/wiki/Death_Star"
]

os.makedirs("raw_pages", exist_ok=True)

for url in PAGES:
    name = url.split("/")[-1]
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")
    with open(f"raw_pages/{name}.txt", "w", encoding="utf-8") as f:
        f.write(text)
