"""
Call nestjs backend to get updated information to update the modal.

The idea is to schedule call this request, process, update the modal.
I can initially retrain from scratch, but should be consider performance concerns at some point.
I then simply upload this pickled model.

Using the following format:
PredictiveModelInputs = {
    sex: 0 | 1, // 0: Female. 1: Male,
    // bmi: number, // body mass index
    body_weight: number // in kg
    height: number // in cm

    // FOOD ITEM (all in grams)
    kilocalories: number,
    carbohydrates: number,
    protein: number,
    fat: number,
    meal_hour: number // int 0-23

    // Previous CGM readings details
    cgm_p30: number,
    cgm_p60: number,
    cgm_p120: number,

    self_identity: SelfIdentity
}
type SelfIdentity = 'Self-identity_African American' | 'Self-identity_Black, African American'| 'Self-identity_Hispanic/Latino' | 'Self-identity_White';
"""
import requests
from datetime import datetime, timedelta
import xgboost as xgb

from model_store.load_model import load_model

BASE_URL = "http://localhost:3000/api/predictive-model/data"

SAMPLE_DATA = {'userHealthData': [{'id': 'cmeev1qtd0001d8444m2cafro', 'userId': 'Wg59wVs3sDakXHFctzRBd1F6Nm1OAWgX', 'gender': 'male', 'dateOfBirth': '2003-08-16T23:00:00.000Z', 'weight': 70, 'height': 170, 'activityLevel': 'moderately_active', 'tdee': 2569, 'proteinTarget': 193, 'carbsTarget': 225, 'fatTarget': 100, 'healthConnected': False, 'createdAt': '2025-08-16T23:00:38.783Z', 'updatedAt': '2025-08-16T23:00:38.783Z'}], 'foodLogData': [{'id': 1, 'userId': 'Wg59wVs3sDakXHFctzRBd1F6Nm1OAWgX', 'foodId': 98, 'quantity': 50.3, 'timestamp': '2025-08-16T23:00:45.737Z', 'food': {'id': 98, 'foodDbSource': 'NZFC', 'NZCompId': 'A1195', 'barcode': None, 'name': 'Bread, from white wheat flour and banana, loaf, as purchased, commercial, composite', 'displayName': 'Banana bread loaf', 'kilocalories': 316, 'carbohydrates': 47, 'sugars': 26.8, 'fat': 10.7, 'fibre': 1.5, 'protein': 7.06, 'alcohol': 0, 'alphaCarotene': 16, 'alphaTocopherol': 1.1, 'ash': 1.7, 'betaCarotene': 31, 'betaTocopherol': 0.13, 'caffeine': 0, 'calcium': 81, 'cholesterol': 0, 'copper': 0.083, 'deltaTocopherol': 0.01, 'dietaryFolateEquivalents': 39, 'dryMatter': 66.9, 'fattyAcidsOmega3': 0.55, 'fattyAcidsMono': 3.69, 'fattyAcidsPoly': 3.21, 'fattyAcidsSaturated': 2.23, 'fattyAcidsTrans': 0.04, 'fibreInsoluble': 1, 'fibreSoluble': 0.5, 'folateTotal': 28, 'fructose': 0.5, 'galactose': 0.2, 'gammaTocopherol': 0.51, 'glucose': 0.6, 'iodide': 28, 'iron': 0.7, 'lactose': 2.2, 'magnesium': 27, 'maltose': 0.1, 'manganese': 370, 'niacinTotal': 1.7, 'nitrogen': 1.13, 'phosphorus': 220, 'potassium': 240, 'retinol': 0, 'riboflavin': 0.2, 'selenium': 7, 'sodium': 410, 'starch': 20.2, 'sucrose': 23.2, 'sugarAdded': 24.5, 'sugarFree': 24.5, 'thiamin': 0.02, 'tryptophan': 72, 'vitaminA': 3, 'vitaminB12': 0, 'vitaminB6': 0.26, 'vitaminC': 0, 'vitaminD': 0, 'vitaminE': 1.2, 'water': 33.1, 'zinc': 0.63}}, {'id': 2, 'userId': 'Wg59wVs3sDakXHFctzRBd1F6Nm1OAWgX', 'foodId': 81, 'quantity': 50.1, 'timestamp': '2025-08-16T23:00:50.042Z', 'food': {'id': 81, 'foodDbSource': 'NZFC', 'NZCompId': 'A1177', 'barcode': None, 'name': 'Bread, from rye flour, sliced, prepacked, as purchased, commercial, composite', 'displayName': 'Rye bread slices', 'kilocalories': 183, 'carbohydrates': 31.5, 'sugars': 3.2, 'fat': 1.86, 'fibre': 9.8, 'protein': 5.18, 'alcohol': 0, 'alphaCarotene': 0, 'alphaTocopherol': 0.52, 'ash': 2.1, 'betaCarotene': 0, 'betaTocopherol': 0.15, 'caffeine': 0, 'calcium': 28, 'cholesterol': 0, 'copper': 0.22, 'deltaTocopherol': 0, 'dietaryFolateEquivalents': 53, 'dryMatter': 54.2, 'fattyAcidsOmega3': 0.14, 'fattyAcidsMono': 0.37, 'fattyAcidsPoly': 0.88, 'fattyAcidsSaturated': 0.29, 'fattyAcidsTrans': 0, 'fibreInsoluble': 7.4, 'fibreSoluble': 2.4, 'folateTotal': 43, 'fructose': 1.1, 'galactose': 0, 'gammaTocopherol': 0.02, 'glucose': 0.7, 'iodide': 0, 'iron': 1.8, 'lactose': 0, 'magnesium': 62, 'maltose': 1.2, 'manganese': 1600, 'niacinTotal': 1.1, 'nitrogen': 0.89, 'phosphorus': 188, 'potassium': 290, 'retinol': 0, 'riboflavin': 0.11, 'selenium': 0, 'sodium': 480, 'starch': 28.3, 'sucrose': 0.2, 'sugarAdded': 0, 'sugarFree': 0, 'thiamin': 0.13, 'tryptophan': 39, 'vitaminA': 0, 'vitaminB12': 0, 'vitaminB6': 0.63, 'vitaminC': 0, 'vitaminD': 0, 'vitaminE': 0.58, 'water': 45.8, 'zinc': 1.66}}], 'mealLogData': [{'id': 1, 'userId': 'Wg59wVs3sDakXHFctzRBd1F6Nm1OAWgX', 'mealId': 1, 'timestamp': '2025-08-16T23:27:09.930Z', 'meal': {'id': 1, 'name': 'My meal', 'userId': 'Wg59wVs3sDakXHFctzRBd1F6Nm1OAWgX', 'mealFoods': [{'id': 1, 'mealId': 1, 'foodId': 4573, 'quantity': 42, 'food': {'id': 4573, 'foodDbSource': 'OFFNZ', 'NZCompId': None, 'barcode': '9421906128024', 'name': 'Eggs', 'displayName': 'Eggs', 'kilocalories': 142, 'carbohydrates': 0.714, 'sugars': 0.714, 'fat': 10, 'fibre': None, 'protein': 12.9, 'alcohol': None, 'alphaCarotene': None, 'alphaTocopherol': None, 'ash': None, 'betaCarotene': None, 'betaTocopherol': None, 'caffeine': None, 'calcium': None, 'cholesterol': None, 'copper': None, 'deltaTocopherol': None, 'dietaryFolateEquivalents': None, 'dryMatter': None, 'fattyAcidsOmega3': None, 'fattyAcidsMono': None, 'fattyAcidsPoly': None, 'fattyAcidsSaturated': None, 'fattyAcidsTrans': None, 'fibreInsoluble': None, 'fibreSoluble': None, 'folateTotal': None, 'fructose': None, 'galactose': None, 'gammaTocopherol': None, 'glucose': None, 'iodide': None, 'iron': None, 'lactose': None, 'magnesium': None, 'maltose': None, 'manganese': None, 'niacinTotal': None, 'nitrogen': None, 'phosphorus': None, 'potassium': None, 'retinol': None, 'riboflavin': None, 'selenium': None, 'sodium': 0.133, 'starch': 0.714, 'sucrose': 0.714, 'sugarAdded': None, 'sugarFree': None, 'thiamin': None, 'tryptophan': None, 'vitaminA': None, 'vitaminB12': None, 'vitaminB6': None, 'vitaminC': None, 'vitaminD': None, 'vitaminE': None, 'water': None, 'zinc': None}}, {'id': 2, 'mealId': 1, 'foodId': 81, 'quantity': 50.1, 'food': {'id': 81, 'foodDbSource': 'NZFC', 'NZCompId': 'A1177', 'barcode': None, 'name': 'Bread, from rye flour, sliced, prepacked, as purchased, commercial, composite', 'displayName': 'Rye bread slices', 'kilocalories': 183, 'carbohydrates': 31.5, 'sugars': 3.2, 'fat': 1.86, 'fibre': 9.8, 'protein': 5.18, 'alcohol': 0, 'alphaCarotene': 0, 'alphaTocopherol': 0.52, 'ash': 2.1, 'betaCarotene': 0, 'betaTocopherol': 0.15, 'caffeine': 0, 'calcium': 28, 'cholesterol': 0, 'copper': 0.22, 'deltaTocopherol': 0, 'dietaryFolateEquivalents': 53, 'dryMatter': 54.2, 'fattyAcidsOmega3': 0.14, 'fattyAcidsMono': 0.37, 'fattyAcidsPoly': 0.88, 'fattyAcidsSaturated': 0.29, 'fattyAcidsTrans': 0, 'fibreInsoluble': 7.4, 'fibreSoluble': 2.4, 'folateTotal': 43, 'fructose': 1.1, 'galactose': 0, 'gammaTocopherol': 0.02, 'glucose': 0.7, 'iodide': 0, 'iron': 1.8, 'lactose': 0, 'magnesium': 62, 'maltose': 1.2, 'manganese': 1600, 'niacinTotal': 1.1, 'nitrogen': 0.89, 'phosphorus': 188, 'potassium': 290, 'retinol': 0, 'riboflavin': 0.11, 'selenium': 0, 'sodium': 480, 'starch': 28.3, 'sucrose': 0.2, 'sugarAdded': 0, 'sugarFree': 0, 'thiamin': 0.13, 'tryptophan': 39, 'vitaminA': 0, 'vitaminB12': 0, 'vitaminB6': 0.63, 'vitaminC': 0, 'vitaminD': 0, 'vitaminE': 0.58, 'water': 45.8, 'zinc': 1.66}}]}}], 'cgmData': []}

def get_data():
    params = {
        "startDate": "2025-08-01T00:00:00",  # ISO format
        "endDate": "2025-08-17T23:59:59"
    }
    req = requests.get(BASE_URL, params=params)
    return req.json()


def process_data(data_json):
    user_health_data = data_json["userHealthData"]
    food_log_data = data_json["foodLogData"]
    meal_log_data = data_json["mealLogData"]
    cgm_data = data_json["cgmData"]

    print(meal_log_data)

    # TEMPORARY: Fixed iAUC
    fixed_iAUC = 150

    xs = []
    ys = []
    for log in food_log_data:
        data = []
        user = find_user(log["userId"], user_health_data)
        gender = 1 if user["gender"] == "male" else 0
        data.extend([gender, user["weight"], user["height"]])

        # ADD NORMALISED MACROS
        data.extend(normalise_macros(log))

        timestamp = datetime.fromisoformat(log["timestamp"].replace("Z", "+00:00"))
        corrected_timestamp = timestamp + timedelta(hours=12)
        data.append(corrected_timestamp.hour)  # Meal hour

        # Add temporary data for cgm values
        data.extend([10, 10, 10])

        data.extend(self_identity_one_hot_encoding('Self-identity_White'))

        xs.append(data)
        ys.append(fixed_iAUC)

    for log in meal_log_data:
        data = []
        user = find_user(log["userId"], user_health_data)
        gender = 1 if user["gender"] == "male" else 0
        data.extend([gender, user["weight"], user["height"]])

        data.extend(normalise_meal_macros(log['meal']['mealFoods']))

        timestamp = datetime.fromisoformat(log["timestamp"].replace("Z", "+00:00"))
        corrected_timestamp = timestamp + timedelta(hours=12)
        data.append(corrected_timestamp.hour)  # Meal hour

        # Add temporary data for cgm values
        data.extend([10, 10, 10])

        data.extend(self_identity_one_hot_encoding('Self-identity_White'))
        xs.append(data)
        ys.append(fixed_iAUC)

    return xs, ys


def normalise_macros(log: dict):
    food = log['food']
    quantity = log['quantity']  # in grams
    data = []
    for key in ['kilocalories', 'carbohydrates', 'protein', 'fat']:
        data.append((quantity / 100) * food[key])
    return data


def normalise_meal_macros(meal_foods: list[dict]):
    data = [0 for i in range(4)]
    for food in meal_foods:
        normalised_macros = normalise_macros(food)
        for i, macro in enumerate(normalised_macros):
            data[i] += macro
    return data


def self_identity_one_hot_encoding(identity: str):
    """
    'Self-identity_African American',
    'Self-identity_Black, African American',
    'Self-identity_Hispanic/Latino',
    'Self-identity_White',
    """
    identities = ['Self-identity_African American',
              'Self-identity_Black, African American',
              'Self-identity_Hispanic/Latino',
              'Self-identity_White']
    index = identities.index(identity)

    one_hot_encoding = []
    for i in range(len(identities)):
        if i == index:
            one_hot_encoding.append(True)
        else:
            one_hot_encoding.append(False)
    return one_hot_encoding


def find_user(user_id, user_data):
    for user in user_data:
        if user["userId"] == user_id:
            return user
    raise Exception("Could not find user.")


def update_modal(model, xs, ys):
    """
    Consider: https://xgboosting.com/update-xgboost-model-with-new-data-using-native-api/
    """
    model.fit(xs, ys, xgb_model=model.get_booster())
    return model



if __name__ == "__main__":
    # json_data = get_data()
    # print(json_data)

    xs, ys = process_data(SAMPLE_DATA)
    # print("Data processed")
    # feature_names, model = load_model("model_v1.pkl")
    # print("Model loaded")
    # new_model = update_modal(model, xs, ys)

