import string
import json
from nltk.stem import PorterStemmer

def tokenize(input):
    punctable = str.maketrans("", "", string.punctuation)
    stemmer = PorterStemmer()
    tokens = input.split()
    tokens = [stemmer.stem(t.lower().translate(punctable)) for t in tokens]
    return tokens

def get_stop_words():
    with open("data/stopwords.txt", "r") as f:
        text = f.read(None)
        return text.splitlines()


def remove_stop_words(l: list[str]):
    stopwords = get_stop_words()
    for stopword in stopwords:
        try:
            l.remove(stopword)
        except Exception:
            continue
    return l


def keyword_search(s: str):
    punctable = str.maketrans("", "", string.punctuation)
    stemmer = PorterStemmer()
    with open("data/movies.json", "r") as f:
        j = json.load(f)
        movies = j["movies"]
        results = []
        tokens = s.split()
        tokens = remove_stop_words(tokens)
        for token in tokens:
            for idx in range(len(movies)):
                item = movies[idx]
                title = item["title"]
                title_tokens = title.split()
                title_tokens = remove_stop_words(title_tokens)
                for title_token in title_tokens:
                    q = stemmer.stem(token.lower().translate(punctable))
                    a = stemmer.stem(title_token.lower().translate(punctable))
                    if q in a:
                        if item not in results:
                            results.append(item)
        results.sort(key=lambda x: x["id"])
        return results
        f.close()


def print_keyword_search(results, limit=5):
    if not results:
        return
    for item in results[:limit]:
        id = item["id"]
        title = item["title"]
        print(f"{id}. {title}")
