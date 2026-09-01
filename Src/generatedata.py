import pandas as pd
import numpy as np
import os

# Set random seed so you get the exact same "random" numbers every time
np.random.seed(42)
n_students = 1000

# 1. Generate the input features
study_hours = np.random.normal(15, 5, n_students).clip(0, 40)
attendance = np.random.normal(85, 10, n_students).clip(0, 100)
previous_score = np.random.normal(70, 12, n_students).clip(0, 100)
assignment_score = np.random.normal(75, 10, n_students).clip(0, 100)
sleep_hours = np.random.normal(7, 1.5, n_students).clip(4, 10)
internet_access = np.random.choice([1, 0], size=n_students, p=[0.85, 0.15])

# 2. Calculate the target variable (Final Score)
# Notice the weights: internet_access gives a flat +5, while sleep_hours multiplies by 1.5
final_score = (
    (study_hours * 0.8) +
    (attendance * 0.3) +
    (previous_score * 0.4) +
    (assignment_score * 0.2) +
    (sleep_hours * 1.5) +
    (internet_access * 5) +
    np.random.normal(0, 4, n_students) # The noise
).clip(0, 100)

# 3. Assemble into a structured DataFrame
df = pd.DataFrame({
    'study_hours': study_hours.round(1),
    'attendance': attendance.round(1),
    'previous_score': previous_score.round(1),
    'assignment_score': assignment_score.round(1),
    'sleep_hours': sleep_hours.round(1),
    'internet_access': internet_access,
    'final_score': final_score.round(1)
})

# 4. Save to the data folder
output_path = 'data/student_performance.csv'
df.to_csv(output_path, index=False)
print(f"Success! Dataset with {n_students} records saved to {output_path}")