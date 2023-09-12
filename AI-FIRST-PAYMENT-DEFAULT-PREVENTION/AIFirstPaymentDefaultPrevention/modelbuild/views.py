from django.shortcuts import render
from django.http import HttpResponse
from rest_framework.decorators import api_view
from rest_framework import viewsets
from django.core import serializers
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from rest_framework.parsers import JSONParser
# Create your views here.

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import joblib

class LoanIncomeExpensesRatioCalculator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['loan_income_expenses_ratio'] = round(X['monthly_payment'] / (X['monthly_income'] - X['monthly_expenses']) * 100, 2)
        return X

class CapitalGenerator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['capital'] = X.apply(self.generate_capital, axis=1)
        return X

    def transform(self, X):
        X['total_capital'] = (X['employment_year'] * 1000000 +
                              X['monthly_income'] * 12 +
                              (X['monthly_income'] - X['monthly_expenses']) * 12 +
                              X['asset_value'] * 0.1)
        return X

class TotalCapitalClassifier(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['clasify_total_capital'] = X.apply(self.classify_total_capital, axis=1)
        return X

    def classify_total_capital(self, X):
        if X['total_capital'] < 100000000:
            return "Sangat Lemah"
        elif X['total_capital'] < 150000000:
            return "Lemah"
        elif X['total_capital'] < 250000000:
            return "Cukup"
        elif X['total_capital'] < 350000000:
            return "Kuat"
        else:
            return "Sangat Kuat"


class DefaultRiskCalculator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['default_risk'] = X.apply(self.calculate_default_risk, axis=1)
        return X

    def calculate_default_risk(self, row):
        if row['loan_income_expenses_ratio'] < 20:
            return "Sangat Baik"
        elif 20 <= row['loan_income_expenses_ratio'] < 40 and row['asset_value'] * 0.2 > row['loan_amount']:
            return "Sangat Baik"
        elif 20 <= row['loan_income_expenses_ratio'] < 40 and row['asset_value'] * 0.5 > row['loan_amount']:
            return "Baik"
        elif 40 <= row['loan_income_expenses_ratio'] < 60 and row['asset_value'] * 0.2 >= row['loan_amount']:
            return "Baik"
        elif 20 <= row['loan_income_expenses_ratio'] < 40 and row['asset_value'] * 0.5 <= row['loan_amount']:
            return "Netral"
        elif 40 <= row['loan_income_expenses_ratio'] < 60 and row['asset_value'] * 0.5 >= row['loan_amount']:
            return "Beresiko"
        elif 40 <= row['loan_income_expenses_ratio'] < 60 and row['asset_value'] * 0.5 <= row['loan_amount']:
            return "Beresiko"
        elif 60 <= row['loan_income_expenses_ratio'] <= 80 and row['asset_value'] * 0.2 >= row['loan_amount']:
            return "Beresiko"
        elif 60 <= row['loan_income_expenses_ratio'] < 80 and row['asset_value'] * 0.5 <= row['loan_amount']:
            return "Sangat Beresiko"
        elif row['loan_income_expenses_ratio'] >= 80:
            return "Sangat Beresiko"
        elif row['loan_income_expenses_ratio'] >= 60:
            return "Beresiko"
        else:
            return None

class SESCalculator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['ses'] = X.apply(self.calculate_ses, axis=1)
        return X

    def calculate_ses(self, row):
        if row['debtor_education_level'] == "SMA":
            if row['monthly_income'] < 3000000 or row['asset_value'] < 50000000 or row['monthly_expenses'] > 0.5 * row['monthly_income']:
                return "Sangat Rendah"
            elif row['monthly_income'] < 8000000 and row['asset_value'] < 400000000 and row['monthly_expenses'] <= 0.6 * row['monthly_income']:
                return "Rendah"
            elif row['monthly_income'] < 15000000 and row['asset_value'] < 800000000 and row['monthly_expenses'] <= 0.7 * row['monthly_income']:
                return "Menengah"
            elif row['monthly_income'] < 30000000 and row['asset_value'] < 1000000000 and row['monthly_expenses'] <= 0.8 * row['monthly_income']:
                return "Tinggi"
            else:
                return "Sangat Tinggi"
        elif row['debtor_education_level'] == "D3" or row['debtor_education_level'] == "D4":
            if row['monthly_income'] < 3500000 or row['asset_value'] < 50000000 or row['monthly_expenses'] > 0.5 * row['monthly_income']:
                return "Sangat Rendah"
            elif row['monthly_income'] < 10000000 and row['asset_value'] < 300000000 and row['monthly_expenses'] <= 0.6 * row['monthly_income']:
                return "Rendah"
            elif row['monthly_income'] < 20000000 and row['asset_value'] < 600000000 and row['monthly_expenses'] <= 0.7 * row['monthly_income']:
                return "Menengah"
            elif row['monthly_income'] < 40000000 and row['asset_value'] < 800000000 and row['monthly_expenses'] <= 0.8 * row['monthly_income']:
                return "Tinggi"
            else:
                return "Sangat Tinggi"
        elif row['debtor_education_level'] == "S1":
            if row['monthly_income'] < 5000000 or row['asset_value'] < 50000000 or row['monthly_expenses'] > 0.5 * row['monthly_income']:
                return "Sangat Rendah"
            elif row['monthly_income'] < 13000000 and row['asset_value'] < 300000000 and row['monthly_expenses']:
                return 'Rendah'
            elif row['monthly_income'] < 26000000 and row['asset_value'] < 600000000 and row['monthly_expenses'] <= 0.7 * row['monthly_income']:
                return "Menengah"
            elif row['monthly_income'] < 52000000 and row['asset_value'] < 1000000000 and row['monthly_expenses'] <= 0.8 * row['monthly_income']:
                return "Tinggi"
            else:
                return "Sangat Tinggi"
        elif row['debtor_education_level'] == "S2":
            if row['monthly_income'] < 7000000 or row['asset_value'] < 50000000 or row['monthly_expenses'] > 0.5 * row['monthly_income']:
                return "Sangat Rendah"
            elif row['monthly_income'] < 15000000 and row['asset_value'] < 400000000 and row['monthly_expenses'] <= 0.7 * row['monthly_income']:
                return "Rendah"
            elif row['monthly_income'] < 30000000 and row['asset_value'] < 800000000 and row['monthly_expenses'] <= 0.8 * row['monthly_income']:
                return "Menengah"
            elif row['monthly_income'] < 60000000 and row['asset_value'] < 1200000000 and row['monthly_expenses'] <= 0.9 * row['monthly_income']:
                return "Tinggi"
            else:
                return "Sangat Tinggi"
        elif row['debtor_education_level'] == "S3":
            if row['monthly_income'] < 9000000 or row['asset_value'] < 50000000 or row['monthly_expenses'] > 0.5 * row['monthly_income']:
                return "Sangat Rendah"
            elif row['monthly_income'] < 18000000 and row['asset_value'] < 600000000 and row['monthly_expenses'] <= 0.7 * row['monthly_income']:
                return "Rendah"
            elif row['monthly_income'] < 36000000 and row['asset_value'] < 1000000000 and row['monthly_expenses'] <= 0.8 * row['monthly_income']:
                return "Menengah"
            elif row['monthly_income'] < 72000000 and row['asset_value'] < 1500000000 and row['monthly_expenses'] <= 0.9 * row['monthly_income']:
                return "Tinggi"
            else:
                return "Sangat Tinggi"
        else:
            return "Tidak Diketahui"  # Tingkat pendidikan tidak dikenali

class DefaultScoreCalculator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['default_score'] = X.apply(self.calculate_default_score, axis=1)
        return X

    def calculate_default_score(self, row):
        default_score = 0

        if row['loan_income_expenses_ratio'] > 70:
            default_score += 500
        elif row['loan_income_expenses_ratio'] > 50:
            default_score += 200
        elif row['loan_income_expenses_ratio'] > 30:
            default_score += 100

        if row['employment_year'] < 1:
            default_score += 250
        elif row['employment_year'] < 3:
            default_score += 100

        if row['loan_purpose'] == "Kebutuhan darurat":
            default_score += 500

        if row['default_risk'] == 'Sangat Beresiko':
            default_score += 1000
        elif row['default_risk'] == 'Beresiko':
            default_score += 750
        elif row['default_risk'] == 'Netral':
            default_score += 500
        elif row['default_risk'] == 'Baik':
            default_score += 250

        if row['asset_value'] * 0.8 <= row['loan_amount']:
            default_score += 300
        elif row['asset_value'] * 0.5 <= row['loan_amount']:
            default_score += 100

        return default_score

class PotentialDefaultCalculator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['default_potential'] = X.apply(self.calculate_default_potential, axis=1)
        return X

    def calculate_default_potential(self, row):
        if row['default_score'] <= 250:
            return 'Sangat Baik'
        elif row['default_score'] <= 500:
            return 'Baik'
        elif row['default_score'] <= 750:
            return 'Netral'
        elif row['default_score'] <= 1250:
            return 'Buruk'
        elif row['default_score'] <= 2000:
            return 'Sangat Buruk'
        else:
            return 'Suspicious'


class SolutionGenerator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['solutions'] = X.apply(self.generate_solution, axis=1)
        return X[['default_score','default_potential','solutions']]

    def generate_solution(self, row):
        solutions = []

        if row['default_potential'] == 'Suspicious':
            solutions.append("Debitur memiliki risiko default yang sangat tinggi, segera lakukan analisis lanjutan")

        if row['debtor_age'] < 25:
            solutions.append("Berikan edukasi keuangan")

        if row['asset_value'] * 0.25 > row['loan_amount']:
            solutions.append("Cairkan aset untuk menutupi pinjaman")
        elif row['asset_value'] * 0.5 > row['loan_amount']:
            solutions.append("Pertimbangkan Cairkan aset untuk menutupi pinjaman")

        if row['loan_purpose'] == "Kebutuhan darurat":
            solutions.append("Penyusunan anggaran untuk kebutuhan mendesak")
        else:
            solutions.append("Evaluasi keuangan, penyusunan anggaran untuk kebutuhan mendesak")

        if row['default_score'] >= 3000:
            solutions.append("Pertimbangkan early write off")
        elif row['default_score'] >= 2000:
            solutions.append("Rencana Pembayaran yang Dapat Dikelola")
        elif row['default_score'] >= 1250:
            solutions.append("Penyederhanaan Kewajiban Keuangan")
        elif row['default_score'] >= 750:
            solutions.append("Konseling Keuangan")

        if row['default_risk'] in ['Sangat Beresiko', 'Beresiko']:
            solutions.append("Prioritaskan Pembayaran")
            solutions.append("Dukungan Emosional")
            solutions.append("Penjadwalan Ulang")

        if row['ses'] == "Rendah":
            solutions.append("Bantuan Keuangan Pemerintah")
        elif row['ses'] == "Menengah":
            solutions.append("Edukasi Keuangan Lanjutan")
        elif row['ses'] == "Tinggi":
            solutions.append("Pengelolaan Keuangan Lanjutan")

        if row['default_risk'] in ['Sangat Beresiko', 'Beresiko'] and row['default_score'] > 2500:
            solutions.append("Opsi pembayaran pokok saja")

        if not solutions:
            return "Nasabah tidak beresiko default."

        return ", ".join(solutions)



from sklearn.base import BaseEstimator, TransformerMixin

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mapping = {
            "debtor_gender": {"laki-laki": 0, "perempuan": 1},
            "debtor_education_level": {
                "SD": 0, "SMP": 1, "SMA": 2, "D3": 3, "D4": 4, "S1": 5, "S2": 6, "S3": 7
            },
            "debtor_marital_status": {
                "Belum menikah": 0, "Sudah menikah": 1, "Cerai Hidup": 2, "Cerai Mati": 3
            },
            "ses": {
                "Sangat Rendah": 0, "Rendah": 1, "Menengah": 2, "Tinggi": 3, "Sangat Tinggi": 4
            },
            "clasify_total_capital": {
                "Sangat Kuat": 1, "Cukup": 2, "Lemah": 3, "Kuat": 4, "Sangat Lemah": 0
            },
            "default_risk": {
                "Baik": 1, "Sangat Baik": 0, "Netral": 2, "Beresiko": 3, "Sangat Beresiko": 4
            },
            "collateral_offered": {"Yes": 0, "No": 1},
            "loan_purpose": {
                "kredit Modal": 1, "Kebutuhan darurat": 2, "kredit pribadi": 3, "pernikahan": 4, "lainnya": 0
            },
            "default_potential" : {"Sangat Baik" : 0,  "Baik" : 1, "Netral" : 2, "Buruk" : 3, "Sangat Buruk" : 4, "Suspicious" : 5}
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for column, mapping in self.mapping.items():
            if column in X_encoded.columns and X_encoded[column].dtype == 'object':
                X_encoded[column] = X_encoded[column].map(mapping)
        return X_encoded



class UselessFeature(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.drop(['debtor_name'], axis=1)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)



def evaluate_model_performance(model, X_test, y_test):
    # Membuat prediksi menggunakan model
    y_pred = model.predict(X_test)

    # Menghitung RMSE
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # Menghitung MAE
    mae = mean_absolute_error(y_test, y_pred)

    # Menghitung R-squared
    ssr = ((y_pred - y_test) ** 2).sum()
    sst = ((y_test - y_test.mean()) ** 2).sum()
    r_squared = 1 - (ssr / sst)

    # Menghitung jumlah data dan jumlah fitur
    n = X_test.shape[0]
    p = X_test.shape[1]

    # Menghitung adjusted R-squared
    adjusted_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))

    return {"mae": round(mae,2),
          "rmse": round(rmse,2),
          "r_squared": round(r_squared,2),
          "adjusted_r_squared": round(adjusted_r_squared,2),}


data_transformation_pipeline = Pipeline([
    ('loan_income_expenses_ratio', LoanIncomeExpensesRatioCalculator()),
    ('capital_generator', CapitalGenerator()),
    ('total_capital_classifier', TotalCapitalClassifier()),
    ('default_risk_calculator', DefaultRiskCalculator()),
    ('ses_calculator', SESCalculator()),
])

target_transformation_pipeline = Pipeline([
    ('default_score_calculator', DefaultScoreCalculator()),
    ('default_potential_calculator', PotentialDefaultCalculator()),
    ('solution_generator', SolutionGenerator())
])


kredit_preprocessing = Pipeline([
    # ('drop_useless_feature', UselessFeature()),
    ('categorical_encoder', CategoricalEncoder()),
    # ('scaler', joblib.load('/content/drive/MyDrive/Magang/model/kredit_pinjaman_scaler.pkl')),
])


solution_preprocessing = Pipeline([
    ('categorical_encoder', CategoricalEncoder()),
    # ('scaler', joblib.load('/content/drive/MyDrive/Magang/model/kredit_pinjaman_solution_scaler.pkl')),
])

model_pipeline = Pipeline([
    ('model', joblib.load('savedmodel/Kredit_pinjaman.pkl'))
])


solution_model = Pipeline([
    ('model', joblib.load("savedmodel/kredit_solution.pkl"))
])

def welcome(request):
    return HttpResponse('Hello World')

import json  

@api_view(["POST"])
def predict(request):
    try:
        mydata = request.data
        unit = list(mydata.values())
        columns = ['debtor_name', 'debtor_gender', 'debtor_age', 'debtor_marital_status', 'number_of_number_of_dependentss', 'debtor_education_level', 'employment_year', 'monthly_income', 'monthly_expenses', 'asset_value', 'collateral_offered', 'loan_amount', 'interest_rate', 'tenor', 'monthly_payment', 'loan_purpose']
        df = pd.DataFrame([unit], columns=columns)
        nama = unit[0]
        df.drop('debtor_name', axis=1, inplace=True)
        data_transformation_pipeline.fit_transform(df)

        new_column_order = ['debtor_gender', 'debtor_age', 'debtor_marital_status',
                            'number_of_number_of_dependentss', 'debtor_education_level',
                            'employment_year', 'monthly_income', 'monthly_expenses', 'asset_value',
                            'collateral_offered', 'loan_amount', 'interest_rate', 'tenor',
                            'monthly_payment', 'loan_income_expenses_ratio', 'total_capital',
                            'clasify_total_capital', 'ses', 'default_risk', 'loan_purpose']

        df = df[new_column_order]

        data_transformation_pipeline.fit_transform(df)
        scaler = joblib.load('savedmodel/kredit_pinjaman_scaler.pkl')
        df = kredit_preprocessing.fit_transform(df)
        scaled_df = scaler.transform(df)
        default_score = model_pipeline.predict(scaled_df)
        df['default_score'] = default_score
        default_potential_cal = PotentialDefaultCalculator()
        default_potential_cal.fit_transform(df)
        solution_df = df[['debtor_age', 'asset_value', 'loan_amount', 'loan_purpose', 'ses', 'default_risk', 'default_score', 'default_potential']]
        solution_df = solution_preprocessing.fit_transform(solution_df)
        solution_scaler = joblib.load('savedmodel/kredit_pinjaman_solution_scaler.pkl')
        solution_df = solution_scaler.transform(solution_df)
        solution_given = solution_model.predict(solution_df)

        # Convert float32 values to Python floats
        default_score = float(df['default_score'].values[0])

        response_data = {
            "nama": nama,
            "default_score": default_score,
            "default_potential": df['default_potential'].values[0],
            "solution": solution_given[0]
        }

        # Serialize the response data to JSON


        return Response(response_data, status = status.HTTP_200_OK)
    except ValueError as e:
        return Response(e.args[0], status=status.HTTP_400_BAD_REQUEST)
