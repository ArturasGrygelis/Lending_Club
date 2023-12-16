FROM python:3.9-slim 
WORKDIR /app
COPY requirements.txt requirements.txt 
RUN pip install --upgrade pip \
    && pip install -r requirements.txt  
COPY ["catcls_subgrade1.pkl","catcls_rejected_accepted_model.pkl","catcls_grade_model.pkl", "app.py", "./"] .
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"] 