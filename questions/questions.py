import nltk
import sys
import os
import math
import string


FILE_MATCHES = 1
SENTENCE_MATCHES = 1

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                content = file.read()
                files[filename] = content
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    tokens = nltk.word_tokenize(document)

    # Remove punctuation and convert to lowercase
    tokens = [token.lower() for token in tokens if token not in string.punctuation]

    # Remove stopwords
    stopwords = set(nltk.corpus.stopwords.words("english"))
    tokens = [token for token in tokens if token not in stopwords]

    return tokens


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = {}
    num_documents = len(documents)

    # Count the number of documents in which each word appears
    word_counts = {}
    for doc_words in documents.values():
        unique_words = set(doc_words)
        for word in unique_words:
            word_counts[word] = word_counts.get(word, 0) + 1

    # Calculate IDF values for each word
    for word, count in word_counts.items():
        idf = math.log(num_documents / count)
        idfs[word] = idf

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    file_scores = {}

    # Calculate tf-idf scores for each file
    for filename, words in files.items():
        score = 0
        for word in query:
            if word in words:
                tf = words.count(word)
                tf_idf = tf * idfs[word]
                score += tf_idf
        file_scores[filename] = score

    top_files = sorted(file_scores, key=file_scores.get, reverse=True)[:n]
    return top_files


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_scores = {}

    # Calculate matching word measure and query term density 
    for sentence, words in sentences.items():
        matching_word_measure = sum(idfs[word] for word in query if word in words)
        query_term_density = sum(word in query for word in words) / len(words)
        sentence_scores[sentence] = (matching_word_measure, query_term_density)
        
    top_sentences = sorted(sentence_scores, key=lambda x: sentence_scores[x], reverse=True)[:n]
    return top_sentences


if __name__ == "__main__":
    main()
