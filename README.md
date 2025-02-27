# stock_prediction
📈 Stock Analysis and Prediction Dashboard
A comprehensive tool for stock market analysis combining traditional technical indicators, machine learning predictions, sentiment analysis, and AI-powered insights.
Overview
This Stock Analysis and Prediction Dashboard provides real-time stock data analysis through multiple approaches:

Technical analysis with candlestick charts and indicators (RSI, MACD)
News sentiment analysis using NLP models
ML-based price predictions with ensemble methods
Deep learning forecasts with LSTM
AI-powered market insights powered by Llama3-70b model
Geopolitical and economic factor analysis

Features
📊 Price Analysis

Interactive candlestick charts
Moving averages and volatility metrics
Technical indicators (RSI, MACD)

📰 News Sentiment

Real-time news retrieval from Yahoo Finance
Sentiment analysis using TextBlob and VADER
FinBERT financial sentiment analysis
Risk score calculation for geopolitical events

🔮 Predictions

Ensemble machine learning predictions using:

Random Forest
XGBoost
Gradient Boosting


Combined ensemble forecasting

🧠 AI Insights

AI-powered market analysis using Llama3-70b via OpenRouter
Structured insights covering:

Price movement analysis
Support/resistance levels
Short and long-term outlooks
Investment strategy recommendations



📈 Advanced Forecasting

LSTM deep learning predictions
5-day price forecasts with visualization

🌍 Market Volatility Analysis

Geopolitical risk index
Political news monitoring
Economic indicators from FRED API (interest rates, inflation, unemployment)

💼 Competitor Analysis

Automatic competitor identification
Comparative price analysis

Technology Stack

Frontend: Streamlit
Data Retrieval:

yfinance for stock data
FRED API for economic data
Yahoo Finance API for news


ML/AI:

scikit-learn (RandomForest, GradientBoosting)
XGBoost
TensorFlow/Keras (LSTM models)
TextBlob & NLTK VADER for sentiment analysis
FinBERT for financial sentiment
Llama3-70b for advanced market insights


Visualization: Plotly

Installation

Clone the repository:

bashCopygit clone https://github.com/yourusername/stock-analysis-dashboard.git
cd stock-analysis-dashboard

Create and activate a virtual environment:

bashCopypython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:

bashCopypip install -r requirements.txt

Set up API keys:

Get a FRED API key from https://fred.stlouisfed.org/docs/api/api_key.html
Get an OpenRouter API key for Llama3-70b access
Update the keys in the code or set as environment variables



Usage

Run the Streamlit app:

bashCopystreamlit run app.py

Enter a stock symbol (e.g., AAPL) and select a time period
Click "Analyze Stock" to generate insights

Research Methodology
Data Sources

Historical stock prices and volumes from Yahoo Finance
News articles from Yahoo Finance and RSS feeds
Economic indicators from Federal Reserve Economic Data (FRED)
Geopolitical risk assessment through news analysis

Models and Algorithms

Sentiment Analysis: Combination of TextBlob, VADER, and FinBERT
Price Prediction: Ensemble of RandomForest, XGBoost, and GradientBoosting
Deep Learning: LSTM networks for time series forecasting
Advanced Analysis: Llama3-70b LLM for comprehensive market insights

Evaluation Metrics

Mean Squared Error for prediction models
Sentiment compound scores for news analysis
Accuracy of support/resistance level identification

Limitations and Future Work

Geopolitical risk index could be enhanced with more sophisticated data sources
Model performance varies based on market volatility and conditions
Limited historical data for some technical indicators
Future work will focus on:

Incorporating option chain data analysis
Adding portfolio optimization features
Implementing risk management tools
Expanding competitor analysis capabilities



Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Federal Reserve Economic Data (FRED) for economic indicators
Yahoo Finance for stock data and news
OpenRouter for LLM API access
The creators of the various machine learning libraries used in this project


####### run the program #######
pip install -r requirements.txt

cd src
streamlit run main.py
