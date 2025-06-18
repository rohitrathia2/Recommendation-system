# 🎬 Movie Recommendation System

This is a Flask-based web application that provides personalized movie recommendations using both **content-based filtering** and **collaborative filtering**. Users can register, log in, rate movies, and receive tailored suggestions based on their preferences and past user activity.

---

## 🚀 Features

- 🔐 **User Authentication** (Register, Login, Logout)
- 🎯 **Hybrid Recommendation Engine**
  - 📚 Content-Based: Recommends movies with similar genres
  - 🤝 Collaborative Filtering: Suggests based on similar users’ ratings
- 📊 **Rating System**: Users can rate movies to improve their recommendations
- 🧠 Built using cosine similarity and TF-IDF for content, and user similarity for collaborative filtering

---

## 🛠 Tech Stack

- **Frontend**: HTML + Jinja2 Templates (for dynamic content rendering)
- **Backend**: Python + Flask
- **Database**: SQLite (`movies.db`)
- **ML Libraries**:
  - `scikit-learn` (TF-IDF, cosine similarity)
  - `pandas`, `numpy`, `scipy`

---

## 📂 Project Structure

├── templates/
│ ├── index.html
│ ├── login.html
│ ├── register.html
│ └── recommendations.html
├── main.py
├── movies.db  


---

## 📥 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/movie-recommendation-system.git
cd movie-recommendation-system


python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows


pip install -r requirements.txt


If requirements.txt is not available, manually install:
pip install flask pandas numpy scikit-learn scipy

python main.py
