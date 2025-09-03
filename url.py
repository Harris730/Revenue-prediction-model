import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def cleanfile():
    alpha=pd.read_csv("my_file.csv",encoding="ISO-8859-1")

    print(alpha.isnull().sum())

    alpha["Peak"]=pd.to_numeric(alpha["Peak"].str.split("[").str[0]) #spilts and return value outside bracket and converts to numeric
    alpha["Peak"]=alpha["Peak"].fillna(alpha["Peak"].mean()) #fill missing values with mean
    alpha["Peak"] = alpha["Peak"].round().astype(int) #rounding off mean values

    alpha["All Time Peak"]=pd.to_numeric(alpha["All Time Peak"].str.split("[").str[0])##
    alpha["All Time Peak"]=alpha["All Time Peak"].fillna(alpha["All Time Peak"].mean())##
    alpha["All Time Peak"] = alpha["All Time Peak"].round().astype(int) ##

    alpha["Ref."]=alpha["Ref."].str.split("[").str[1].str.replace("]","").astype(int) #just using value inside brackets

    alpha["Actualgross"]=alpha["Actualgross"].str.split("[").str[0].str.replace("]","") #removing index with brackets
    alpha["Actualgross"] = alpha["Actualgross"].replace('[\\$,]', '', regex=True).astype(float) #removing $ and , from data and convert to float

    alpha["Adjustedgross (in 2022 dollars)"] = alpha["Adjustedgross (in 2022 dollars)"].replace('[\\$,]', '', regex=True).astype(float)#removing $ and , from data and convert to float

    alpha[["Artist","Tour title"]] = alpha[["Artist","Tour title"]].apply(lambda col: col.str.replace(r'[^A-Za-z\s]', '', regex=True)) #just using alphabets

    alpha["Average gross"] = alpha["Average gross"].replace('[\\$,]', '', regex=True).astype(float)#removing $ and , from data and convert to float
    alpha["Averagegross adjusted"]=(alpha["Adjustedgross (in 2022 dollars)"]/alpha["Shows"]).round(0)

    alpha["Year(s)"]=alpha["Year(s)"].str.replace("â","-") # replacing junks with -

    #print(alpha)
    alpha.to_csv("real_data.csv", index=False, encoding="utf-8-sig")

def Z_factor(col):
    mean_x=col.mean()
    deviation=(col-mean_x)**2
    Standard=(deviation.sum()/len(col))**0.5 #standard deviation
    z_factor=(col-mean_x)/Standard #zfactor
    #print("zfactor : \n",z_factor)
    return z_factor
   # minmax= (col-col.min())/(col.max()-col.min())
  #print("Minmax :",minmax)
  
def scaling(df):
  df = pd.DataFrame(cod)
# Standard Scaler
  stanscaler = StandardScaler()
  df["Shows"] = stanscaler.fit_transform(df[["Shows"]])
  X = df[["Shows"]]         
  y = df["Average gross"]    
  x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=0.8, random_state=42)
  model=LinearRegression()
  model.fit(x_train,y_train)
  #print(predicted_gross)
  y_pred=model.predict(x_test)

  mae=mean_squared_error(y_test,y_pred)
  print("MAE:",round(mae,2))
  r2=r2_score(y_test,y_pred)
  print("R2:",round(r2,2))

  # alpha=int(input("Enter shows :"))
  # alpha_scaled = stanscaler.transform([[alpha]])
  # predic=model.predict(alpha_scaled)
  # print("Averge gross is :",round(predic[0],2))

  plt.scatter(x_test, y_test, color="blue", label="Actual")
  plt.plot(x_test, y_pred, color="red", linewidth=2, label="Predicted Line")
  plt.xlabel("Shows")
  plt.ylabel("Average gross")
  plt.title("Linear Regression Fit")
  plt.legend()
  plt.show()
    # predicted_gross=model.predict(X)
  # print("x_train:\n", x_train)
  # print("x_test:\n", x_test)
  # print("y_train:\n", y_train)
  # print("y_test:\n", y_test)

def regression(cod):
    model = LinearRegression()
    Y=cod[["Shows"]]
    X=cod["Average gross"]
    model.fit(X,Y)
    show= float(input("Enter Shows : "))
    predict_gross= model.predict([[show]])
    print(f"You enter {show} so your Gross will be {predict_gross}")
   
def visualization(df):
    Y=df["Shows"]
    X=df["Average gross"]
    plt.scatter(X, Y, color="blue", marker="o")
    plt.xlabel("Shows")
    plt.ylabel("Average Gross")
    plt.plot()
    plt.show()

def confus_metrics():
    y_true=[1,0,0,1,1,0,0,1,1,1]
    y_pred=[1,0,1,0,0,1,0,1,1,0]
    cm=confusion_matrix(y_true,y_pred)
    print(cm)

def removing_outliers(thresh,cod):
    # Loop through each numeric column
    for col in cod.select_dtypes(include=["int64", "float64"]).columns:
        cod[f"z_{col}"] = Z_factor(cod[col])   # add z-score column
        cod = cod[cod[f"z_{col}"].abs() <= thresh]  # filter rows

    return cod
    # cod["z_gross"]=Z_factor(col)
    # cod=cod[cod["z_gross"].abs()<=3]
    # return cod

if __name__ == "__main__": 
    
    cod=pd.read_csv("expanded_data.csv")
    threshold=2.5
    cod=removing_outliers(threshold,cod)
    #print(cod)
    scaling(cod)
    # X=cod[["Shows"]]
    # Y=cod["Average gross"]
    # model=LinearRegression()
    # model.fit(X,Y)
    # predicted_gross=model.predict(X)
    # mae=mean_squared_error(Y,predicted_gross)
    # mse=mean_squared_error(Y,predicted_gross)
    # rmse=np.sqrt(mse)
    # r2=r2_score(Y,predicted_gross)

    # print("MAE:",round(mae,2))
    # print("MSE:",round(mse,2))
    # print("RMSE:",round(rmse,2))
    # print("R2:",round(r2,2))
    # plt.figure(figsize=(10,6))
    # plt.scatter(X,Y,color="blue",label="Actual score")
    # plt.plot(X,predicted_gross,color="red",label="line")
    # plt.grid(True)
    #plt.show()
    alpha=5+5
    # # # #new_pred=model.predict([[400]])
    #print("predicted gross :",new_pred)



