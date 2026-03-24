from infer_api import predict_implicit_aspects

text = "I am a 100 lb girl, had a glass of wine and a glass of beer prior to the dinner, and I was still HUNGRY after my visit to this place!"
print(predict_implicit_aspects(text, domain="electronics", top_k=5))