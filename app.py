import streamlit as st
import json
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from nltk import FreqDist
from nltk.chunk import ne_chunk


@st.cache_data
def morphological(sent: str):
    tokens = word_tokenize(sent)
    tagged_tokens = pos_tag(tokens)
    morph = []
    for token, pos in tagged_tokens:
        morph.append((token, pos))
    return morph


@st.cache_data
def semantic_entity(text: str):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    named_entities = ne_chunk(tagged_tokens)
    sem_en = []
    for entity in named_entities.subtrees():
        if entity.label() != 'S':
            sem_en.append(' '.join(word for word, pos in entity.leaves()), entity.label())
    return sem_en


@st.cache_data
def lexical(sent: str):
    tokens = word_tokenize(sent)
    lexi = []
    for token in tokens:
        lexi.append(("Token:", token))
    return lexi


@st.cache_data
def syntax(sent: str):
    tokens = word_tokenize(sent)
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    pos_tags = pos_tag(tokens)
    stop_words = get_stop_words('en')
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return list(zip(tokens, lemmas, [tag for _, tag in pos_tags], [tag for _, tag in pos_tags], filtered_tokens))


@st.cache_data
def semantic_similar_words(sentences):
    # Split sentences into words
    sentences = sentences.split("\n")
    sentences = [word_tokenize(sentence) for sentence in sentences]

    # Remove stopwords and perform part-of-speech tagging
    stop_words = set(stopwords.words('english'))
    tagged_sentences = []
    for sentence in sentences:
        tagged_sentence = pos_tag(sentence)
        filtered_sentence = [(word, tag) for word, tag in tagged_sentence if word.lower() not in stop_words]
        tagged_sentences.append(filtered_sentence)

    # Create WordNet similarity matrix
    similarity_matrix = []
    for tagged_sentence in tagged_sentences:
        synsets = []
        for word, tag in tagged_sentence:
            if tag.startswith('N'):
                synset = wordnet.synsets(word, pos=wordnet.NOUN)
            elif tag.startswith('V'):
                synset = wordnet.synsets(word, pos=wordnet.VERB)
            elif tag.startswith('J'):
                synset = wordnet.synsets(word, pos=wordnet.ADJ)
            elif tag.startswith('R'):
                synset = wordnet.synsets(word, pos=wordnet.ADV)
            else:
                synset = []
            synsets.extend(synset)
        similarity_matrix.append(synsets)

    # Calculate most similar words
    similar_words = {}
    for i, synsets in enumerate(similarity_matrix):
        fdist = FreqDist()
        for synset in synsets:
            for lemma in synset.lemmas():
                fdist[lemma.name()] += 1
        similar_words[sentences[i]] = fdist.most_common(10)

    return str(similar_words)


@st.cache_data
def discourse(sentences):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Remove stopwords
    stopwords_list = get_stop_words('en')
    sentences_without_stopwords = [[word for word in sentence.split() if word.lower() not in stopwords_list] for
                                   sentence in sentences]

    disc = []
    # Perform discourse analysis
    for i, sentence in enumerate(sentences_without_stopwords):
        disc.append(f"Sentence {i + 1}: {' '.join(sentence)}")
        if i < len(sentences_without_stopwords) - 1:
            disc.append(f"Next sentence: {' '.join(sentences_without_stopwords[i + 1])}")
        disc.append("-" * 50)
    return disc


@st.cache_data
def pragmatic(sentence):
    # Tokenize the sentence
    tokens = word_tokenize(sentence)

    # Perform part-of-speech tagging
    tagged_tokens = pos_tag(tokens)

    # Apply named entity recognition (NER)
    ner_tags = ne_chunk(tagged_tokens)

    prag = []
    # Analyze the pragmatics of named entities
    for subtree in ner_tags.subtrees():
        if subtree.label() == 'GPE':  # Filter for geographical entities
            entity = ' '.join(word for word, tag in subtree.leaves())
            context = ' '.join(word for word, _ in subtree.sent)
            prag.append(f"Entity: {entity}\nLabel: {subtree.label()}\nContext: {context}" + "-" * 50)


st.title("NLP - Analyzer📝")

text = st.text_input("Input String")
st.write("OR")


def content_to_dict(content):
    # Convert content to dictionary
    data = {"file_contents": content}
    return data


def read_txt_file(file):
    content = file.read().decode('utf-8')
    return content


# Create a file uploader button
uploaded_file = st.file_uploader("Upload a .txt file", type=".txt")

# Check if a file is uploaded
if uploaded_file is not None:
    # Read the contents of the file
    text = read_txt_file(uploaded_file)

if st.checkbox("Show File Contents"):
    st.write("File Contents:")
    st.write(text)


if text or uploaded_file:
    st.title("Morphological Analysis")
    data = morphological(text)
    st.write(data)
    data = content_to_dict(data)
    file_name = uploaded_file.name + "_morphological_analysis.json"
    json_data = json.dumps(data, indent=4)
    st.download_button("Download File", json_data, file_name=file_name)
    st.title("Lexical Analysis")
    data = lexical(text)
    st.write(data)
    data = content_to_dict(data)
    file_name = uploaded_file.name + "_lexical_analysis.json"
    json_data = json.dumps(data, indent=4)
    st.download_button("Download File", json_data, file_name=file_name)
    st.title("Syntax Analysis")
    syntax_result = syntax(text)
    for token, lemma, pos_tag, _, filtered_token in syntax_result:
        st.write("Token:", token)
        st.write("Lemma:", lemma)
        st.write("POS Tag:", pos_tag)
        st.write("Filtered Token:", filtered_token)
        st.write("-" * 50)
    data = []
    for token, lemma, pos_tag, _, filtered_token in syntax_result:
        item = {
            "Token": token,
            "Lemma": lemma,
            "POS Tag": pos_tag,
            "Filtered Token": filtered_token
        }
        data.append(item)
    json_data = json.dumps(data, indent=4)
    file_name = uploaded_file.name + "_syntax_analysis.json"
    st.download_button("Download File", json_data, file_name=file_name)

    # st.title("Semantic (entity) Analysis")
    # st.write(semantic_entity(text))
    # st.title("Semantic (similar words) Analysis")
    # similar_words = semantic_similar_words(text)
    # similar_words = eval(similar_words)  # Convert string back to dictionary
    # for sentence, words in similar_words.items():
    #     st.write("Similar words for:", sentence)
    #     for word, count in words:
    #         st.write(word, count)
    #     st.write()
    # st.title("Pragmatic Analysis")
    # st.write(pragmatic(text))
