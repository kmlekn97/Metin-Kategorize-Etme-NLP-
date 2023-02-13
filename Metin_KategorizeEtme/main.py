import pandas as pd
from sklearn import preprocessing
from DataProcessService import DataProcessService
from ModelTuning import ModelTuning
def findKategori(result):
    if result==0:
        return "Dünya"
    elif result==1:
        return "Ekonomi"
    elif result==2:
        return "Kültür"
    elif result==3:
        return "Sağlık"
    elif result==4:
        return "Siyaset"
    elif result==5:
        return "Spor"
    else:
        return "Teknoloji"
df = pd.read_csv("7allV03.csv")
encoder = preprocessing.LabelEncoder()
y = encoder.fit_transform(df.category)
metinisle=DataProcessService()
metinisle.metinIsle(df,'text')
texts = []
texts=metinisle.createText(df)
model=ModelTuning()
model.CreateCountVectorizer(texts,y,5000)
model.ModelLGBMClassifier("countvectorizer + lgbm:")
model.ModelXGBClassifier("countvectorizer + xgb:")
model.CreateSampleMeanTfidfVectorizer(texts,y,5000)
model.ModelXGBClassifier("Tfid vec + lgbm")
model.ModelXGBClassifier("Tfid vec + xgb")
model.yazdir()
print("Güven Aralığı:",model.enbuyuk()[1])
yeni_yorum=" Ekonomi altının onsu 1 750 dolar sınırında seyrediyor uluslararası piyasada altının onsu geçtiğimiz hafta gerçekleştirdiği sert yükselişin ardından bugün sınırlı da olsa satıcılı seyrediyor ve 1 750 dolar sınırından işlem görüyor altının onsu geçtiğimiz haftanın son işlem günü yurt içi piyasaların kapanışa yakın saatlerde sert bir şekilde yükselişe geçerek 1 729 dolardan başladığı günü 1 752 dolara kadar yükselmişti analistler altının anlık olarak 1 750 dolar direncinin üzerini görse de bu seviyenin kırıldığından bahsetmek için henüz erken olduğunu belirtiyor teknik analistler altının onsunda 1 750 dolar direncinin kırılması durumunda ters omuz baş omuz tobo formasyonunun teyit edileceğini dolayısıyla stop loss alış emirlerinin gelebileceğini kaydediyor bu durumda altının yükselişine devam etmesinin beklenebileceğini ifade eden analistler sonraki hedeflerin 1 775 ve 1 790 seviyeleri olabileceğini tahmin ediyor"
model.CreateCountVectorizer(texts,y,len(model.findMaxNode(yeni_yorum,0)[0]))
model.ModelLGBMClassifier("countvectorizer + lgbm:")
model.ModelXGBClassifier("countvectorizer + xgb:")
model.CreateSampleMeanTfidfVectorizer(texts,y,len(model.findMaxNode(yeni_yorum,0)[0]))
model.ModelXGBClassifier("Tfid vec + lgbm")
model.ModelXGBClassifier("Tfid vec + xgb")
model.yazdir()
print("Güven Aralığı:",model.enbuyuk()[1])
model.SolveModel(model.enbuyuk()[0],texts,y,df,yeni_yorum)
print(findKategori(model.ModelResult()))