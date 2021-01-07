# this file only runs with the 3.6.9 + intepreter (Python 3.8)
import aspect_based_sentiment_analysis as absa


# establish a pipeline for the model (including a reviewer)
name = 'absa/classifier-rest-0.2'
model = absa.BertABSClassifier.from_pretrained(name)
tokenizer = absa.BertTokenizer.from_pretrained(name)
professor = absa.Professor()
text_splitter = absa.sentencizer()

# Break
nlp = absa.Pipeline(model, tokenizer, professor=[])

text = ("We are great fans of Slack, but we wish the subscriptions "
        "were more accessible to small startups.")

slack, price = nlp(text, aspects=["slack", "price"])


# what does slack.scores mean??
print(slack.sentiment)
# assert price.sentiment == absa.Sentiment.negative
# assert slack.sentiment == absa.Sentiment.positive

