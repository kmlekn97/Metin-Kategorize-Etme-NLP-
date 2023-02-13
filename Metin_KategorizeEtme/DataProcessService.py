from DataProcess import DataProcess
class DataProcessService(DataProcess):
    def __init__(self):
        pass

    def tokenizasyon(self, text):
        return super().tokenizasyon(text)

    def convert_lowercase(self, text):
        return super().convert_lowercase(text)

    def remove_punctuation(self, text):
        return super().remove_punctuation(text)

    def remove_stopwords(self, text):
        return super().remove_stopwords(text)

    def remove_numbers(self, text):
        return super().remove_numbers(text)

    def remove_less_than_2(self, text):
        return super().remove_less_than_2(text)

    def remove_extra_space(self, text):
        return super().remove_extra_space(text)

    def metinIsle(self,df,area):
        df[area] = df[area].apply(self.convert_lowercase)
        df[area] = df[area].apply(self.remove_punctuation)
        df[area] = df[area].apply(self.remove_stopwords)
        df[area] = df[area].apply(self.remove_extra_space)
        df[area] = df[area].apply(self.remove_numbers)
        df[area] = df[area].apply(self.remove_less_than_2)
    def createText(self,df):
        texts = []
        for text in df.text:
            texts.append(text)
        return texts
