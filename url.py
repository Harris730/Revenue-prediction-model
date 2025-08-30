import pandas as pd
#from sklearn.preprocessing import LabelEncoder

def cleanfile():
    alpha=pd.read_csv("my_file.csv",encoding="ISO-8859-1")

    print(alpha.isnull().sum())

    alpha["Peak"]=pd.to_numeric(alpha["Peak"].str.split("[").str[0]) #spilts and return value outside bracket and converts to numeric
    alpha["Peak"]=alpha["Peak"].fillna(alpha["Peak"].mean()) #fill missing values with mean
    alpha["Peak"] = alpha["Peak"].round().astype(int) #rounding off mean values

    alpha["All Time Peak"]=pd.to_numeric(alpha["All Time Peak"].str.split("[").str[0])##
    alpha["All Time Peak"]=alpha["All Time Peak"].fillna(alpha["All Time Peak"].mean())##
    alpha["All Time Peak"] = alpha["All Time Peak"].round().astype(int) ##

    alpha["Ref."]=alpha["Ref."].str.split("[").str[1].str.replace("]","") #just using value inside brackets

    alpha["Actualgross"]=alpha["Actualgross"].str.split("[").str[0].str.replace("]","") #removing index with brackets
    alpha["Actualgross"] = alpha["Actualgross"].replace('[\\$,]', '', regex=True).astype(float) #removing $ and , from data and convert to float

    alpha["Adjustedgross (in 2022 dollars)"] = alpha["Adjustedgross (in 2022 dollars)"].replace('[\\$,]', '', regex=True).astype(float)#removing $ and , from data and convert to float

    alpha[["Artist","Tour title"]] = alpha[["Artist","Tour title"]].apply(lambda col: col.str.replace(r'[^A-Za-z\s]', '', regex=True)) #just using alphabets

    alpha["Average gross"] = alpha["Average gross"].replace('[\\$,]', '', regex=True).astype(float)#removing $ and , from data and convert to float
    alpha["Averagegross adjusted"]=(alpha["Adjustedgross (in 2022 dollars)"]/alpha["Shows"]).round(0)

    alpha["Year(s)"]=alpha["Year(s)"].str.replace("â","-") # replacing junks with -

    print(alpha)
    alpha.to_csv("cleaned_data.csv", index=False, encoding="utf-8-sig")


def encoding():
  pd.set_option("display.max_columns", None)
  df_cod=pd.read_csv("cleaned_data.csv")
  label=pd.get_dummies(df_cod,columns=["Artist"],dtype=int)

  print(label)


if __name__ == "__main__": 
 encoding()


