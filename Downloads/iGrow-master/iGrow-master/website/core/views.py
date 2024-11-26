from django.shortcuts import render
from .forms import CropRecommendationForm, DiseasePredictionForm, PestIdentificationForm
import os
import joblib

# Path to the saved model file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'static/ml_models/classifier_rf.joblib')


model_path = MODEL_PATH
model = joblib.load(model_path)

# Create your views here.
def home(request):
    user = request.user
    context = {
        "title": "Home",
        "content": "Welcome to the home page!",
        "user": user,
    }

    return render(request, "core/index.html", context=context)


def disease(request):

    context = {
        "current_page": "disease",
    }
    if request.method == "POST":
        form = DiseasePredictionForm(request.POST, request.FILES)
        if form.is_valid():
            # Process the form data
            image = form.cleaned_data["image"]

            # Make a prediction
            prediction_result = "apple scab"
            confidence = 80

            # Save the result (optional)
            # PredictionResult.objects.create(
            #     result_type='disease',
            #     result=prediction_result,
            #     confidence=confidence
            # )

            context.update(
                {
                    "predicted_result": {
                        "result": prediction_result,
                        "confidence": confidence,
                    },
                }
            )

    else:
        form = DiseasePredictionForm()

    context.update({"form": form})

    return render(request, "core/disease.html", context=context)


def crop_recommendation(request):

    context = {
        "current_page": "crop_recommendation",
    }
    prediction_result = None
    if request.method == "POST":
        form = CropRecommendationForm(request.POST)
        if form.is_valid():
            # Process the form data
            nitrogen = form.cleaned_data["nitrogen"]
            phosphorus = form.cleaned_data["phosphorus"]
            potassium = form.cleaned_data["potassium"]
            temperature = form.cleaned_data["temperature"]
            humidity = form.cleaned_data["humidity"]
            ph = form.cleaned_data["ph"]
            # soil_moisture = form.cleaned_data["soil_moisture"]
            rainfall = form.cleaned_data["rainfall"]

            # Create input array for the model
            input_data = [[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]]

            # Predict the crop
            prediction_result = model.predict(input_data)

            context.update(
                {
                    "form": form,
                    "predicted_result": {
                        "result": prediction_result,
                    },
                }
            )
    else:
        form = CropRecommendationForm()
        context.update({"form": form})

    return render(request, "core/crop-recommendation.html", context=context)


def pest(request):

    context = {
        "current_page": "pest_identification",
    }
    if request.method == "POST":
        form = PestIdentificationForm(request.POST, request.FILES)
        if form.is_valid():
            # Process the form data
            image = form.cleaned_data["image"]

            # Make a prediction
            prediction_result = "apple scab"

            context.update(
                {
                    "form": form,
                    "predicted_result": {
                        "result": prediction_result,
                    },
                }
            )

    else:
        form = PestIdentificationForm()
        context.update({"form": form})

    return render(request, "core/pest.html", context=context)
