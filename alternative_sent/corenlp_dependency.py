from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pycorenlp import StanfordCoreNLP
import json

nlp = StanfordCoreNLP('https://corenlp.run/')

target = "bread"
text = "this bread is tasty and warm but the sauce is too cold"


def find_related(arr, keyword):

    deps = []
    for node in arr:
        if node["dependentGloss"] == keyword:
            deps.append(node)

    return deps

def main():

    # gets an output
    output = nlp.annotate(text, properties={
        'annotators': 'tokenize,ssplit,pos,depparse',
        'outputFormat': 'json'
    })

    print(output)

    # store the enhanced++ dep
    enhanced = output["sentences"][0]["enhancedPlusPlusDependencies"]

    targeted = find_related(enhanced, target)


    with open("res.json", 'w') as f:
        json.dump(targeted, f)


main()