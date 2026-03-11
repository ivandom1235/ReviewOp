# How to Run the Project

## Backend

# The following code works in python 3.13.

1. cd backend  
2. python -m venv venv  
3. venv\Scripts\activate  
4. pip install -r requirements.txt  
5. Go to backend\core\config.py. Here you need to change MySQL user and password for the backend to work.
6. python -m uvicorn app:app --host 127.0.0.1 --port 8000  
7. go to http://127.0.0.1:8000/docs   
8. go to post/infer/review  
9. try it out  

## Frontend

1. cd frontend  
2. npm install  
3. npm run dev  
4. try it out  