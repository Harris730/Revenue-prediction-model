import pandas as pd

alpha=pd.read_csv("my_file.csv",encoding="ISO-8859-1")

#print(alpha.isnull().sum())

#alpha=alpha.dropna()
#print(alpha[["Peak","All Time Peak"]])

# alpha["Peak"]=pd.to_numeric(alpha["Peak"].str.split("[").str[0])
# alpha["Peak"].fillna(alpha["Peak"].mean(),inplace=True)
# alpha["Peak"] = alpha["Peak"].round().astype(int)

# alpha["All Time Peak"]=pd.to_numeric(alpha["All Time Peak"].str.split("[").str[0])
# alpha["All Time Peak"].fillna(alpha["All Time Peak"].mean(),inplace=True)
# alpha["All Time Peak"] = alpha["All Time Peak"].round().astype(int)

alpha["Ref."]=alpha["Ref."].str.split("[").str[1].str.replace("]","")
alpha["Tour title"]=alpha["Tour title"].str.split("[").str[0].str.replace("]","")
alpha["Year(s)"]=alpha["Year(s)"].str.replace("â","-")

alpha[["Artist","Tour title"]] = alpha[["Artist","Tour title"]].apply(lambda col: col.str.replace(r'[^A-Za-z\s]', '', regex=True))

alpha["Year(s)"] = alpha["Year(s)"].astype(str).str.rjust(12)

print(alpha[["Artist","Tour title","Year(s)"]])
#print(alpha["Actual gross"])
#print(alpha)
#print(alpha)
