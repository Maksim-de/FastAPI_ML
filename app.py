import pandas as pd
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
from fastapi.encoders import jsonable_encoder
import pickle
import sklearn
import io
import numpy as np
from fastapi.responses import StreamingResponse

## Получение модели и encoder
with open('model.pikle', 'rb') as f:
    file = pickle.load(f)
loaded_model = file['model']
loaded_encoder = file['encoder']


## Функция для преобразования столбцов mileage,	engine,	max_power
def data_value(x):
    if type(x)==str:
        x = x.split()
        try:
            x = float(x[0])
            return x
        except ValueError:
            return np.nan
    else:
        return float(x)

# Удаление и замена столбцов
def new_prepare_col(dataframe, columns : list):
    for i in columns:
        dataframe.loc[:, i] = dataframe[i].apply(data_value)
        dataframe[i] = dataframe[i].astype(float)
    return dataframe

def onehot(dataframe, columns : list):
    df_ohe = dataframe[columns]
    new_column_names = loaded_encoder.get_feature_names_out()
    ## Получим датасет с закодированными столбцами
    df_ohe = pd.DataFrame(
        loaded_encoder.transform(df_ohe),
        columns=new_column_names,
        index=df_ohe.index  # Используем тот же индекс, что и у исходного DataFrame
    )
    ## Отберем некатегориальные признаки
    df_real = dataframe.select_dtypes(include=['int', 'float']).copy()
    df_real.drop(columns='seats', inplace=True)
    ## Объединим таблицы
    dataframe = df_real.join(df_ohe)
    return dataframe

# Предсказание данных
def predict(data_prepare):
    return loaded_model.predict(data_prepare)

# Преобразование данных
def prepr(data):
    print(data)
    data = new_prepare_col(data, ['engine',	'max_power', 'mileage'])
    data = data.drop(columns=['name', 'torque'])
    data = onehot(data, ['fuel', 'seller_type', 'transmission', 'owner', 'seats'])
    return data



app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: int

def Filling_empty(data):
    col = data.columns
    for i in col:
        data[i].fillna(data[i].median(), inplace = True)
    return data

class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    return predict(prepr(pd.DataFrame([jsonable_encoder(item)])))

@app.post("/predict_items")
async def modify_csv(file: UploadFile):
    contents = await file.read()
    contents_str = contents.decode('utf-8')
    data_csv = pd.read_csv(io.StringIO(contents_str), sep=',')
    data = prepr(data_csv)
    data = Filling_empty(data)
    y = predict(data)
    data_csv['selling_price'] = y
    csv_data = data_csv.to_csv(index=False)
    return StreamingResponse(
            csv_data,
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=modified_file.csv"
            }
        )

# uvicorn main:app --reload --port 8000
# app - приложение FastAPI()
# main - название файла