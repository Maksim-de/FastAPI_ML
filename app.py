import pandas as pd
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel, Field
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
    name: str = Field(example="Maruti Swift Dzire VDI")
    year: int = Field(example=2014)
    km_driven: int = Field(example=145500)
    fuel: str = Field(example="Diesel")
    seller_type: str = Field(example="Individual")
    transmission: str = Field(example="Manual")
    owner: str = Field(example="First Owner")
    mileage: str = Field(example="23.4 kmpl")
    engine: str = Field(example="1248 CC")
    max_power: str = Field(example="74 bhp")
    torque: str = Field(example="190Nm@ 2000rpm")
    seats: int = Field(example=5)

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
