from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app= FastAPI()
@app.get("/BMI")
def BMI(weight:float , hight:float):
   bmi = weight / (hight **2)
   if bmi < 18.5:
         return {"BMI": bmi, "Status": "Underweight"}
   elif 18.5 <= bmi < 24.9:
         return {"BMI": bmi, "Status": "Normal weight"}
   elif 25 <= bmi < 29.9:
         return {"BMI": bmi, "Status": "Overweight"}
   else:
         return {"BMI": bmi, "Status": "Obesity"}