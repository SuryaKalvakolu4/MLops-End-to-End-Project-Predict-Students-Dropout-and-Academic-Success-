# app/schema.py

from pydantic import BaseModel
from typing import Optional

class PredictionRequest(BaseModel):
    Marital_status: str
    Application_mode: int
    Application_order: int
    Course: int
    Daytime_evening_attendance: int
    Previous_qualification: int
    Previous_qualification_grade: Optional[float] = None
    Nationality: int
    Mother_qualification: int
    Father_qualification: int
    Mother_occupation: int
    Father_occupation: int
    Admission_grade: float
    Displaced: int
    Educational_special_needs: int
    Debtor: int
    Tuition_fees_up_to_date: int
    Gender: str
    Scholarship_holder: int
    Age_at_enrollment: int
    International: int
    Curricular_units_1st_sem_enrolled: float
    Curricular_units_1st_sem_evaluations: float
    Curricular_units_1st_sem_approved: float
    Curricular_units_1st_sem_grade: float
    Curricular_units_1st_sem_without_evaluations: float
    Curricular_units_2nd_sem_enrolled: float
    Curricular_units_2nd_sem_evaluations: float
    Curricular_units_2nd_sem_approved: float
    Curricular_units_2nd_sem_grade: float
    Curricular_units_2nd_sem_without_evaluations: float
    Unemployment_rate: float
    Inflation_rate: float
    GDP: float
