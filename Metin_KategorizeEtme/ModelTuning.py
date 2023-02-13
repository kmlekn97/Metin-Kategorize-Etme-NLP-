from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split

class ModelTuning():
    __Accuracy=[]
    __method=[]
    __X_train=None
    __X_test=None
    __y_train=None
    __y_test=None
    __modeltn=None
    def CreateSampleMeanTfidfVectorizer(self,texts,y,max):
        tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=max)
        X = tfidf.fit_transform(texts).toarray()
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    def CreateCountVectorizer(self,texts,y,max):
        cv = CountVectorizer(max_features=max)
        X = cv.fit_transform(texts).toarray()
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    def findMaxNode(self,yorum,vector):
        if vector==0:
            v=CountVectorizer()
        else:
            v=TfidfVectorizer()
        snc = v.fit_transform([yorum]).toarray()
        return snc
    def ModelLGBMClassifier(self,type):
        lgbm = LGBMClassifier()
        lgbm.fit(self.__X_train, self.__y_train)
        y_pred_lgbm = lgbm.predict(self.__X_test)
        print("Accuracy of ",type, accuracy_score(self.__y_test, y_pred_lgbm))
        print("Precision of ",type, precision_score(self.__y_test, y_pred_lgbm, average="micro"))
        accuracy_score
        self.__Accuracy.append(accuracy_score(self.__y_test, y_pred_lgbm))
        self.__method.append(type)
    def ModelXGBClassifier(self,type):
        xgb = XGBClassifier()
        xgb.fit(self.__X_train, self.__y_train)
        y_pred_xgb = xgb.predict(self.__X_test)
        print("Accuracy of ",type, accuracy_score(self.__y_test, y_pred_xgb))
        print("Precision of ", type , precision_score(self.__y_test, y_pred_xgb, average="micro"))
        self.__Accuracy.append(accuracy_score(self.__y_test, y_pred_xgb))
        self.__method.append(type)
    def enbuyuk(self):
        data=self.__Accuracy[0]
        indis=0
        for i in range(len(self.__Accuracy)):
            if self.__Accuracy[i] > data:
                data=self.__Accuracy[i]
                indis=i
        return indis,data
    def SolveCountVector(self,yorum,y,df,texts):
        v = CountVectorizer()
        snc = v.fit_transform([yorum]).toarray()
        cv = CountVectorizer(max_features=len(snc[0]))
        X = cv.fit_transform(texts).toarray()
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(df['text'], y, test_size=0.2, random_state=101)
        return cv
    def SolveTfidfVectorizer(self,yorum,y,df,texts):
        t = TfidfVectorizer()
        snc = t.fit_transform([yorum]).toarray()
        tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=len(snc[0]))
        X = tfidf.fit_transform(texts).toarray()
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(df['text'], y, test_size=0.2,random_state=101)
        return tfidf

    def ChoiceModel(self, model, yorum,y,df,texts,vector):
        x_train_count = vector.transform(self.__X_train).toarray()
        X_test = vector.transform(self.__X_test).toarray()
        model.fit(x_train_count, self.__y_train)
        snc = vector.fit_transform([yorum]).toarray()
        self.__modeltn=int(model.predict(snc)[0])
    def SolveModel(self,indis,texts,y,df,yorum):
        if indis==0:
            vector=self.SolveCountVector(yorum,y,df,texts)
            lgbm=LGBMClassifier()
            self.ChoiceModel(lgbm,yorum,y,df,texts,vector)
        elif indis==1:
            xgb=XGBClassifier()
            vector=self.SolveCountVector(yorum,y,df,texts)
            self.ChoiceModel(xgb, yorum, y, df, texts,vector)
        elif indis==2:
            lgbm = LGBMClassifier()
            vector = self.SolveTfidfVectorizer(yorum, y, df, texts)
            self.ChoiceModel(lgbm, yorum, y, df, texts,vector)
        else:
            xgb=XGBClassifier()
            vector=self.SolveTfidfVectorizer(yorum,y,df,texts)
            self.ChoiceModel(xgb, yorum, y, df, texts,vector)

    def yazdir(self):
        print(self.__Accuracy)
        print(self.__method)
    def ModelResult(self):
        return self.__modeltn