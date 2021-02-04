from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
sent = "Don't get captured, or you'll get fucked"
words = word_tokenize(sent)
bigrams = Counter(zip(words, words[1:]))

print(bigrams)

# with open('reddit_sarcasm.txt') as file:
#     sentences = file.readlines()

# haha = {}

# for i, sentence in enumerate(sentences):
#     print(i)
#     for token in word_tokenize(sentence): 
#         haha[token] = haha.get(token, 0) + 1


# nltk.download('punkt')
# nltk.download()