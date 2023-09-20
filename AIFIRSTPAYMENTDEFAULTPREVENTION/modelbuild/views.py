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
import json  

from preprocessing.kredit import data_transformation_pipeline, kredit_preprocessing, \
    solution_preprocessing, model_pipeline, solution_model, PotentialDefaultCalculator

def welcome(request):
    return HttpResponse('Hello World')



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
            "default_score": round(default_score),
            "default_potential": df['default_potential'].values[0],
            "solution": solution_given[0],
            "status" : status.HTTP_200_OK
        }

        # Serialize the response data to JSON


        return Response(response_data, status = status.HTTP_200_OK)
    except ValueError as e:
        error_message = {
            "error" : str(e),
            "status": status.HTTP_400_BAD_REQUEST
        }
        return Response(error_message, status=status.HTTP_400_BAD_REQUEST)
