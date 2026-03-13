# How to Run the Project

> **Python Version**: This project requires Python 3.13+

## Quick Start with Automated Scripts (Recommended)

### Option 1: One-Time Setup + Manual Run
Run the setup script once to install all dependencies:
```powershell
.\run-project.ps1
```

Then start services in separate terminals:
```powershell
# Terminal 1 - Backend
cd backend
venv\Scripts\activate
python -m uvicorn app:app --host 127.0.0.1 --port 8000

# Terminal 2 - Frontend
cd frontend
npm run dev
```

### Option 2: Automatic Service Launch (Quicker Method)
After setup, run both services with one command:
```powershell
.\run-services.ps1
```
This will open new terminal windows for both backend and frontend.

## Manual Setup Instructions

### Backend

1. cd backend  
2. python -m venv venv  
3. venv\Scripts\activate  
4. pip install -r requirements.txt  
5. **Configure Database**: Go to `backend\core\config.py` and update MySQL username and password
6. python -m uvicorn app:app --host 127.0.0.1 --port 8000  
7. Visit http://127.0.0.1:8000/docs (Swagger UI)
8. Test the `/infer/review` POST endpoint

### Default Seeded Login Accounts

On backend startup, the app seeds two accounts if they do not already exist:

- Admin
  - username: `admin`
  - password: `12345`
- User
  - username: `user`
  - password: `12345`

Login page is available in the frontend at `/login`.

- Admin users are routed to `/admin`
- Normal users are routed to `/`

### Frontend

1. cd frontend  
2. npm install  
3. npm run dev  
4. Follow the URL shown in the terminal  

## Graph Views

- Single review explanation graph:
  1. Start backend and frontend.
  2. Submit a review in the `Single Review` form.
  3. Open the `Review Explanation` graph mode on the dashboard.

- Batch / corpus aspect graph:
  1. Upload a CSV in the `Batch CSV` form.
  2. Switch to `Corpus Analytics` graph mode.
  3. Use the graph filter bar to change `domain`, `product_id`, date range, and minimum edge weight.

## Graph API

- `GET /graph/review/{review_id}` returns the single-review explanation graph.
- `GET /graph/aspects` returns the batch co-occurrence graph.
