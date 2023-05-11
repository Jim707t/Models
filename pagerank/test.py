from pagerank import sample_pagerank, crawl
import sys
import json
import random


x = [44, 65, 646, 7, 74, 82]
y = [0.1, 0.2, 0.3, 0.1, 0.2, 0.1]
print(type(random.choices(x, weights=y)))