#  Crypto Prophet Dashboard

> **Real-Time Market Forecasting Engine**
> An end-to-end Data Science portfolio project integrating a vanilla frontend with a high-performance FastAPI backend to predict Cryptocurrency, Forex, and Commodity markets.

A full-stack machine learning application designed to forecast the prices of top-tier assets (including Bitcoin, Ethereum, Gold, and EUR/USD). The system pulls live market data via YFinance and utilizes Lasso Regression—optimized for minimizing Mean Squared Error (MSE) against volatile financial noise—to generate real-time predictive insights.

---

##  Table of Contents
- [Architecture](#-architecture)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Installation & Execution](#-installation--execution)
- [Model Evaluation](#-model-evaluation)
- [Disclaimer](#-disclaimer)

---

##  Architecture
The project adopts a decoupled architecture:
1. **Frontend:** A high-fidelity, responsive, and framework-less UI built with HTML/CSS/JS. It captures user parameters (asset and forecast horizon) and renders dynamic SVG trend charts based on API responses.
2. **Backend:** A fast, asynchronous API powered by FastAPI. It handles incoming requests, fetches live market history, dynamically trains the selected machine learning model, and returns the forecast payload.

---

##  Key Features
* **Live Market Data:** Direct integration with Yahoo Finance for zero-latency historical price retrieval.
* **Multi-Asset Support:** Covers Top 10 Cryptocurrencies, Gold futures (GC=F), and Forex (EUR/USD).
* **Noise Reduction:** Leverages L1 regularization (Lasso Regression) to aggressively prune noisy features inherent in volatile markets.
* **Instant Forecasting:** On-the-fly model training and inference based on user-defined forecast periods (Daily, Weekly, Monthly).
* **Interactive Data Viz:** Custom-built inline SVG charting that plots historical data against the predicted trajectory without relying on heavy external libraries.

---

##  Tech Stack
**Frontend:**
* HTML5, CSS3 (Custom properties, No frameworks)
* Vanilla JavaScript (ES6+, Fetch API)

**Backend & Machine Learning:**
* Python 3
* FastAPI & Uvicorn
* Scikit-Learn (Lasso Regression)
* XGBoost
* Pandas & NumPy
* YFinance

---

##  Installation & Execution

### Prerequisites
Ensure Python is installed on your system. Install the required backend dependencies:
```bash
pip install fastapi uvicorn yfinance pandas numpy xgboost scikit-learn
```

### Running the Application

**1. Start the Backend API**

Navigate to the project directory and launch the FastAPI server:
```bash
uvicorn main:app --reload
```
The server will start at `http://localhost:8000`.

**2. Launch the Frontend**

Open `index.html` in any modern web browser. (You can double-click the file or use a local server like VS Code's Live Server).

---

##  Model Evaluation
During the development phase, candidate models were evaluated on financial datasets. Due to the extreme volatility and noise in financial time-series data, complex ensemble models like XGBoost demonstrated a tendency to overfit short-term fluctuations.

Lasso Regression was ultimately selected as the primary predictive engine due to its L1 penalty, which effectively filters out noise and captures the underlying fundamental trend, resulting in the most optimal Mean Squared Error (MSE) during validation.

---

##  Disclaimer
This project is an educational Data Science portfolio piece. The predictive models are experimental and rely on simplified drift-and-noise assumptions validated against historical data. This application does not provide financial advice and should not be used to make real-world trading, purchasing, or investment decisions.

---

**Developed by Dimas Arya Ramadhan** | Data Science, Institut Teknologi Sumatera (ITERA)