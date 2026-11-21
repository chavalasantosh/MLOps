import bentoml
import bentoml.mlflow
import pandas as pd
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

YesNo = Literal["Yes", "No"]

class PatientInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: float = Field(ge=1, le=120, allow_inf_nan=False)
    gender: Literal["Male", "Female"]
    polyuria: YesNo
    polydipsia: YesNo
    sudden_weight_loss: YesNo
    weakness: YesNo
    polyphagia: YesNo
    genital_thrush: YesNo
    visual_blurring: YesNo
    itching: YesNo
    irritability: YesNo
    delayed_healing: YesNo
    partial_paresis: YesNo
    muscle_stiffness: YesNo
    alopecia: YesNo
    obesity: YesNo

@bentoml.service
class DiabetesClassifier:
    model_ref = bentoml.models.BentoModel('diabetes_risk_educational:boucwoxfq2zkeasc')

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.model_ref)

    @bentoml.api
    def predict(self, input_data: PatientInput) -> dict:
        frame = pd.DataFrame([input_data.model_dump()])
        frame["age"] = frame["age"].astype("float64")
        result = int(self.model.predict(frame)[0])
        return {"prediction": result, "label": "Positive" if result == 1 else "Negative",
                "model": 'SVM-Educational', "registry_version": '1'}

    @bentoml.api
    def version(self) -> dict:
        return {
            "application": "Diabetes MLOps",
            "version": "1.1.0",
            "framework": "BentoML",
            "purpose": "Educational"
        }
