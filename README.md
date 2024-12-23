**Предсказательная модель стоимости автомобилей**

**FastApi** https://fastapi-ml-19qx.onrender.com/docs#/default/predict_item_predict_item_post

Json для тестирования API *predict_item* хранится в файле **test.txt**\
CSV для тестирования API *predicts_item* хранится в файле **car.csv**

Этот проект представляет собой разработку и развертывание модели машинного обучения для предсказания стоимости автомобилей.
Было исследовано несколько моделей регрессии, чтобы определить оптимальное решение для данной задачи.

В рамках проекта были реализованы следующие модели регрессии:

Линейная регрессия: Базовая модель, используемая в качестве эталона.
Lasso регрессия: Модель с L1-регуляризацией, способствующая занулению маловажных признаков.
Ridge регрессия: Модель с L2-регуляризацией.
ElasticNet регрессия: Комбинация Lasso и Ridge регрессий, позволяющая настраивать баланс между L1 и L2 регуляризациями.
Для каждой модели был проведен подбор гиперпараметров с помощью GridSearchCV, что позволило оптимизировать их производительность и выбрать наилучшие настройки. 
Качество моделей оценивалось на основе [R-квадрата, MSE и реализованной бизнес-метрике: доля прогнозов, отличающихся от реальных цен на авто не более чем на 10% (в одну или другую сторону)]. 

Результаты сравнения моделей представлены ниже:
*LinearRegression_r2: 0.5941419794788385*\
**business_metric: 0.227** 

*LinearRegression_St_r2: 0.5941419794788517*\
**business_metric: 0.227**

*Lasso_r2: 0.5850819172869055*\
**business_metric: 0.232**

*ElasticNet_r2: 0.5885307104796541*\
**business_metric: 0.233**

*Ridge_r2: 0.6448193398814482*\
**business_metric: 0.248**

Лучшее качество показала Ridge - регрессия

Развертывание сервиса:

Для удобного использования разработанной модели создан API с использованием фреймворка FastAPI. Сервис позволяет отправлять запросы с входными данными и получать предсказания от обученной модели.\
Сервис был развернут при помощи https://render.com/.

Ключевые технологии:
Python\
Scikit-learn\
Pandas\
FastAPI

Низкое качество моделей объясняется игнорированием некоторых столбцов датасета, в которых хранится множество информации, характеризующей автомобили.
