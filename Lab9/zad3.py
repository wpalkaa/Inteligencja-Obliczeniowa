# snscrape nie działa, dawno nieaktualizowany i coś tam nie działa już
import requests

subreddit = "witcher"  
limit = 100
url = f"https://www.reddit.com/r/{subreddit}/new.json?limit={limit}"

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Safari/1.0'}

print(f"Pobieranie {limit} postów z r/{subreddit}...")
response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    posts = data['data']['children']
    
    with open('reddit_posts.txt', 'w', encoding='utf-8') as f:
        for i, post in enumerate(posts, 1):
            title = post['data'].get('title', '')
            selftext = post['data'].get('selftext', '').replace('\n', ' ')
            f.write(f"[{i}] TYTUŁ: {title}\n")
            f.write(f"TREŚĆ: {selftext}\n")
            f.write("-" * 50 + "\n")
            
    print("Pomyślnie zapisano posty do pliku reddit_posts.txt")
else:
    print(f"Błąd pobierania danych. Kod statusu: {response.status_code}")