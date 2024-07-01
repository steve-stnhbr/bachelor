import requests

word_site = "https://www.mit.edu/~ecprice/wordlist.10000"
response = requests.get(word_site)
WORDS =  [str(line) for line in response.content.splitlines()]

def get_word(i: int):
    i = i % len(WORDS)
    return WORDS[i]

def get_random_word():
    import random
    return get_word(random.randint(0, len(WORDS) - 1))