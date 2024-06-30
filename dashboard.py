import pandas as pd
import streamlit as st
import requests

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'data': data}
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()

def main():
    MLFLOW_URI = "http://127.0.0.1:5000/invocations"

    st.sidebar.title('Paramètres')
    st.title('Prédiction de Score de Crédit')

    EXT_SOURCE_1 = st.number_input('EXT_SOURCE_1',
                                   min_value=0.0, value=0.52, step=1.0)

    EXT_SOURCE_2 = st.number_input('EXT_SOURCE_2',
                                   min_value=0.0, value=0.60, step=1.0)

    EXT_SOURCE_3 = st.number_input('EXT_SOURCE_3',
                                   min_value=0.0, value=0.57, step=1.0)

    DAYS_EMPLOYED = st.number_input('DAYS_EMPLOYED',
                                    min_value=0.0, value=0.8, step=1.0)

    DAYS_BIRTH = st.number_input('DAYS_BIRTH',
                                 min_value=0.0, value=0.48, step=100.0)

    client_installments_AMT_PAYMENT_min_sum = st.number_input('client_installments_AMT_PAYMENT_min_sum',
                                                              min_value=0.0, value=0.04, step=1.0)

    bureau_DAYS_CREDIT_max = st.number_input('bureau_DAYS_CREDIT_max',
                                             min_value=0.0, value=0.84, step=1.0)

    bureau_DAYS_CREDIT_ENDDATE_max = st.number_input('bureau_DAYS_CREDIT_ENDDATE_max',
                                                     min_value=0.0, value=0.62, step=1.0)

    client_cash_CNT_INSTALMENT_FUTURE_mean_max = st.number_input('client_cash_CNT_INSTALMENT_FUTURE_mean_max',
                                                                 min_value=0.0, value=0.17, step=1.0)

    OWN_CAR_AGE = st.number_input('OWN_CAR_AGE',
                                  min_value=0.0, value=0.11, step=1.0)

    bureau_AMT_CREDIT_SUM_DEBT_mean = st.number_input('bureau_AMT_CREDIT_SUM_DEBT_mean',
                                                      min_value=0.0, value=0.02, step=1.0)

    DAYS_ID_PUBLISH = st.number_input('DAYS_ID_PUBLISH',
                                      min_value=0.0, value=0.58, step=1.0)

    predict_btn = st.button('Prédire')

    if predict_btn:
        data = [[EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, DAYS_EMPLOYED, DAYS_BIRTH,
                 client_installments_AMT_PAYMENT_min_sum, bureau_DAYS_CREDIT_max,
                 bureau_DAYS_CREDIT_ENDDATE_max, client_cash_CNT_INSTALMENT_FUTURE_mean_max,
                 OWN_CAR_AGE, bureau_AMT_CREDIT_SUM_DEBT_mean, DAYS_ID_PUBLISH]]
        try:
            pred = request_prediction(MLFLOW_URI, data)[0] * 100000
            st.write('Le score du client est {:.2f}'.format(pred))
        except Exception as e:
            st.write("Erreur lors de la prédiction: {}".format(e))

if __name__ == '__main__':
    main()
