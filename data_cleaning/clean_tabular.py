import numpy as np
import pandas as pd
import re
from spellchecker import SpellChecker
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

class CleanTabular():
    def __init__(self, path): 
        pd.options.mode.chained_assignment = None
        self.path = path
        self.df = pd.read_csv(self.path, lineterminator='\n')
        pd.set_option('display.max_columns', None)

    def drop_nulls(self, df, **kwargs):
        if 'subset' in kwargs:
            subset = kwargs['subset']
            df.dropna(subset=subset)
        else:
            df.dropna()
        return df

    def drop(self, df, **kwargs):
        if 'column_names' in kwargs:
            for column in kwargs['column_names']:
                df.drop([column], axis=1, inplace=True)
        if 'row_indexes' in kwargs:
            for index in kwargs['row_indexes']:
                df.drop(index, inplace=True)
        return df
    
    def split_column(self, df, column, delimiter, **kwargs):
        if 'return_columns' in kwargs:
            split_df = pd.DataFrame(index=np.arange(len(df.index)), columns=['empty'])
            for col in kwargs['return_columns']:
                temp_df = df[column].str.split(delimiter, expand=True)[col]
                split_df = pd.concat([split_df.reset_index(drop=True), temp_df.reset_index(drop=True)], axis=1)
            split_df = self.drop(split_df, column_names=['empty'])
            return split_df
        else:
            return df[column].str.split('|', expand=True)
    
    def df_contains_rows(self, df, column, contains):
        df = df[df[column].str.contains(contains)]
        df.reset_index(drop=True, inplace=True)
        return df
    
    def replace(self, df, column, replacee, replacer):
        return df[column].apply(lambda x: x.replace(replacee, replacer))
    
    def clean_title(self, df):
        df['gumtree'] = self.split_column(df, column='product_name', delimiter='|', return_columns=[2])
        df = self.df_contains_rows(df, column='gumtree', contains='Gumtree')
        df = self.drop(df, column_names=['gumtree'])
        df['product_name'] = self.split_column(df, column='product_name', delimiter='|', return_columns=[0])
        return df

    def clean_price(self, df):
        df['price'] = self.replace(df, column='price', replacee='£', replacer='')
        df['price'] = self.replace(df, column='price', replacee=',', replacer='')
        df['price'] = df['price'].astype(float)
        return df
    
    def values_split(self, df, column):
        return df[column].apply(lambda x: x.split())
    
    def values_join(self, df, column):
        return df[column].apply(lambda x: ' '.join(x))
    
    def remove_from_values_regex(self, df, column, regex, replacer):
        return df[column].apply(lambda x: [re.sub(r'{}'.format(regex), replacer, y) for y in x])
    
    def remove_from_end(self, df, column, removee):
        return df[column].apply(lambda x: [y.replace(removee, '') if y.find(removee) == len(y)-1 else y for y in x])
    
    def remove_unwanted_chars(self, df, column):
        df['temp_col'] = self.values_split(df, column)
        df['temp_col'] = self.remove_from_values_regex(df, 'temp_col', '[^0-9A-Za-z.+ ]',  ' ')
        df['temp_col'] = self.remove_from_end(df, 'temp_col', '.')
        df[column] = self.values_join(df, 'temp_col')
        df = self.drop(df, column_names=['temp_col'])
        return df
    
    def decapitalisation(self, df, column):
        return df[column].apply(lambda x: x.lower())
    
    def remove_stops(self, df, column):
        stops = set(stopwords.words("english"))
        stops.add("price")
        df[column] = df[column].apply(lambda x: [y for y in x if y not in stops])
        return df[column]

    def remove_stop_words(self, df, column):
        df['temp_col'] = self.values_split(df, column)
        df['temp_col'] = self.remove_stops(df, 'temp_col')
        df[column] = self.values_join(df, 'temp_col')
        df = self.drop(df, column_names=['temp_col'])
        return df
    
    def lemmatise(self, df, column):
        lemmatizer = nltk.stem.WordNetLemmatizer()
        return df[column].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])
    
    def lemmatise_text(self, df, column):
        df['temp_col'] = self.values_split(df, column)
        df['temp_col'] = self.lemmatise(df, 'temp_col')
        df[column] = self.values_join(df, 'temp_col')
        df = self.drop(df, column_names=['temp_col'])
        return df

    def clean_text(self, df, columns):
        for column in columns:
            df = self.remove_unwanted_chars(df, column)
            df[column] = self.decapitalisation(df, column)
            df = self.lemmatise_text(df, column)
            df = self.remove_stop_words(df, column)
        return df

    def clean(self):
        self.df = self.df.replace(['N/A'], np.NaN)
        self.df = self.drop_nulls(self.df, subset=['product_name', 'category', 'product_description', 'location', 'price'])
        self.df = self.clean_title(self.df)
        self.df = self.clean_price(self.df)
        self.df = self.clean_text(self.df, ['product_name', 'product_description', 'location'])

    def save(self):
        pass

if __name__ == '__main__':
    clean = CleanTabular('./data/Products.csv')
    clean.clean()