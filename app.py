import pandas as pd
from flask import Flask, render_template, request, url_for, redirect
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
import re
from nltk.corpus import stopwords
import string

app = Flask(__name__)








nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
stopword=set(stopwords.words('english'))
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text









def sentiment_score(a, b, c):
    if (a>b) and (a>c):
        out="Positive 😊 "
    elif (b>a) and (b>c):
        out="Negative 😠 "
    else:
        out="Neutral 🙂 "
    return(out)



@app.route("/", methods=['POST', 'GET'])
def funct():

    product = [1,2,3,4,5]

    return render_template("home.html",product=product)


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    input_data = request.form.get('Product')
    prod=input_data

    prodname= request.form.get('ProductName')
    Rev = request.form.get('Review')
    data = pd.read_csv("C:/Users/Admin/PycharmProjects/Flipkart1/myapp/Data/Data.csv")
    df1 = {
            'Product_name': [prodname],
            'Review': [Rev],
            'Rating': [prod]
        }
    data1 = pd.DataFrame(df1)
    data1.to_csv('C:/Users/Admin/PycharmProjects/Flipkart1/myapp/Data/Data.csv', mode='a', index=False, header=False)
    data = data[data.Product_name == prodname]
    data["Review"] = data["Review"].apply(clean)
    nltk.download('stopwords')
    stemmer = nltk.SnowballStemmer("english")
    stopword = set(stopwords.words('english'))
    nltk.download('vader_lexicon')
    sentiments = SentimentIntensityAnalyzer()
    data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["Review"]]
    data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["Review"]]
    data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["Review"]]
    data = data[["Review", "Positive", "Negative", "Neutral"]]
    x = float(sum(data["Positive"]))
    y = float(sum(data["Negative"]))
    z = float(sum(data["Neutral"]))
    x = float("{:.2f}".format(x))
    y = float("{:.2f}".format(y))
    z = float("{:.2f}".format(z))
    print(prodname)
    print(prod)
    print(Rev)
    sent=sentiment_score(x, y, z)
    print(data.head())
    return render_template("home.html",variable=sent,Neutral='Neutral',Positive='Positive',Negative='Negative',Neutral1=z,Positive1=x,Negative1=y)









'''
data = pd.read_csv("flipkart_reviews.csv")
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")

stopword = set(stopwords.words('english'))


def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text


data["Review"] = data["Review"].apply(clean)
ratings = data["Rating"].value_counts()
numbers = ratings.index
quantity = ratings.values
nltk.download('vader_lexicon')
sentiments = SentimentIntensityAnalyzer()
data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["Review"]]
data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["Review"]]
data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["Review"]]
data = data[["Review", "Positive", "Negative", "Neutral"]]
x = sum(data["Positive"])
y = sum(data["Negative"])
z = sum(data["Neutral"])


def sentiment_score(a, b, c):
    if (a > b) and (a > c):
        out = "Positive ?? "
    elif (b > a) and (b > c):
        out = "Negative ?? "
    else:
        out = "Neutral ?? "
    return (out)


sentiment_score(x, y, z)
'''
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=30006, debug=True)
