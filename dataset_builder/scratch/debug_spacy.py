import spacy
nlp = spacy.load("en_core_web_sm")
text = "The battery life is amazing but the screen is too dim."
doc = nlp(text)
for token in doc:
    print(f"{token.text:12} {token.pos_:10} {token.dep_:10} {token.head.text:12}")
