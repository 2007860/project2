# Data Analyst Agent - Final (TDS)

## Run locally with Python
1. python -m venv venv
2. source venv/bin/activate   # Windows: venv\\Scripts\\activate
3. pip install --upgrade pip
4. pip install -r requirements.txt
5. uvicorn app.main:app --host 0.0.0.0 --port 8000
6. curl -X POST "http://localhost:8000/api/" -F "questions.txt=@test_client/example_questions.txt" -m 180
