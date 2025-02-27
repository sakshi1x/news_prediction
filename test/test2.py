from venv import logger
import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
import yfinance as yf
from bs4 import BeautifulSoup
import requests
import feedparser
from datetime import datetime, timedelta
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import warnings
from fredapi import Fred  # For economic data
from test import  AdvancedStockAnalyzer, calculate_technical_indicators

warnings.filterwarnings('ignore')

nltk.download('vader_lexicon', quiet=True)

# FRED API key (get one from https://fred.stlouisfed.org/docs/api/api_key.html)
FRED_API_KEY = '5a24dcab75c5f77ff277b689babdc8da'

class StockAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.openrouter_key = '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' 
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        self.scaler = StandardScaler()
        self.models = {
            'RandomForest': RandomForestRegressor(n_estimators=100),
            'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100)
        }
        self.fred = Fred(api_key=FRED_API_KEY)  # Initialize FRED API
    def generate_insights(self, context):
        """Generate insights using Llama3-70b via OpenRouter"""
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "HTTP-Referer": "https://your-domain.com",
            "X-Title": "Stock Analyzer"
        }
        
        prompt = f"""Analyze this stock context and provide detailed insights:
        {context}
        
        Structure your response with:
        1. Price Movement Reasons (Technical & Fundamental)
        2. Key Support/Resistance Levels
        3. Short-term Outlook (1-5 days)
        4. Long-term Considerations
        5. Investor Strategy Recommendations
        """
        
        data = {
            "model": "meta-llama/llama-3-70b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.6,
            "max_tokens": 1200
        }
        
        try:
            response = requests.post(self.openrouter_url, headers=headers, json=data)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            return "Error: Failed to get valid response from AI model"
        except Exception as e:
            return f"Connection Error: {str(e)}"
        
    

    def get_geopolitical_risk_index(self):
        """Fetch real geopolitical risk index from an API."""
        # Placeholder: Replace with a real API call
        return np.random.uniform(0, 1)  # Simulated risk index

    def get_economic_data(self):
        """Fetch real economic data (interest rate, inflation, unemployment) from FRED."""
        try:
            interest_rate = self.fred.get_series('FEDFUNDS').iloc[-1]  # Federal Funds Rate
            inflation = self.fred.get_series('CPIAUCSL').pct_change(12).iloc[-1] * 100  # Inflation rate
            unemployment_rate = self.fred.get_series('UNRATE').iloc[-1]  # Unemployment rate
            return interest_rate, inflation, unemployment_rate
        except Exception as e:
            print(f"Error fetching economic data: {e}")
            return 0.05, 0.02, 0.04  # Fallback values

    def get_news(self, symbol, news_type='general'):
        """Fetch news from different sources based on type."""
        news_items = []
        
        if news_type == 'general':
            # Fetch news from Yahoo Finance
            try:
                stock = yf.Ticker(symbol)
                yahoo_news = stock.news
                for item in yahoo_news:
                    news_items.append({
                        'title': item['title'],
                        'text': item.get('summary', ''),
                        'timestamp': datetime.fromtimestamp(item['providerPublishTime']),
                        'type': 'general'
                    })
            except Exception as e:
                print(f"Error fetching Yahoo news: {e}")

            # Fetch news from RSS feeds
            try:
                rss_feed = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US'
                feed = feedparser.parse(rss_feed)
                for entry in feed.entries:
                    news_items.append({
                        'title': entry.title,
                        'text': entry.summary,
                        'timestamp': datetime.now(),  # RSS feeds may not always have a timestamp
                        'type': 'general'
                    })
            except Exception as e:
                print(f"Error fetching RSS: {e}")

        elif news_type == 'political':
            # Fetch political/geopolitical news from Google News RSS
            try:
                political_rss = f'https://news.google.com/rss/search?q={symbol}+(politics|geopolitical|government|regulation|war|sanctions)&hl=en-US&gl=US&ceid=US:en'
                feed = feedparser.parse(political_rss)
                for entry in feed.entries[:10]:  # Limit to 10 articles
                    news_items.append({
                        'title': entry.title,
                        'text': entry.description,
                        'timestamp': datetime(*entry.published_parsed[:6]),
                        'type': 'political'
                    })
            except Exception as e:
                print(f"Error fetching political news: {e}")

        return news_items

    def analyze_sentiment(self, text):
        """Analyze sentiment of a given text using TextBlob and VADER."""
        # TextBlob sentiment analysis
        analysis = TextBlob(text)
        textblob_polarity = analysis.sentiment.polarity  # Polarity score from TextBlob

        # VADER sentiment analysis
        vader_score = self.vader.polarity_scores(text)['compound']  # Compound score from VADER

        # Combine scores (average of TextBlob and VADER)
        combined_score = (textblob_polarity + vader_score) / 2

        # Custom geopolitical risk keywords
        risk_keywords = ['war', 'sanction', 'embargo', 'conflict', 'tariff', 'treaty']
        risk_score = sum(1 for word in risk_keywords if word in text.lower()) * 0.1

        return {
            'compound': combined_score,  # Combined sentiment score
            'risk_score': min(risk_score, 1.0)  # Cap risk score at 1.0
        }

    def get_competitor_prices(self, symbol):
        """Fetch real competitor stock prices using Yahoo Finance."""
        competitors = {
            'AAPL': ['MSFT', 'GOOGL', 'AMZN'],  # Example competitors for Apple
            'MSFT': ['AAPL', 'GOOGL', 'AMZN'],  # Example competitors for Microsoft
            # Add more mappings as needed
        }
        competitor_prices = {}
        for comp in competitors.get(symbol, []):
            try:
                stock = yf.Ticker(comp)
                hist = stock.history(period="1d")
                competitor_prices[comp] = hist['Close'].iloc[-1]
            except Exception as e:
                print(f"Error fetching competitor {comp} data: {e}")
                competitor_prices[comp] = 0  # Fallback value
        return competitor_prices

    def prepare_features(self, data, symbol):
        """Prepare features with real data."""
        # Existing features
        data['price_change'] = data['Close'].pct_change().fillna(0)
        data['volume_change'] = data['Volume'].pct_change().fillna(0)
        data['moving_avg_5'] = data['Close'].rolling(window=5).mean().fillna(0)
        data['moving_avg_10'] = data['Close'].rolling(window=10).mean().fillna(0)
        data['volatility'] = data['Close'].rolling(window=5).std().fillna(0)

        # Political/geopolitical features
        data['geopolitical_risk'] = self.get_geopolitical_risk_index()

        # Economic features
        interest_rate, inflation, unemployment_rate = self.get_economic_data()
        data['interest_rate'] = interest_rate
        data['inflation'] = inflation
        data['unemployment_rate'] = unemployment_rate

        # Competitor prices
        competitor_prices = self.get_competitor_prices(symbol)
        for comp, price in competitor_prices.items():
            data[f'competitor_{comp}_price'] = price

        # Scale the features
        features_to_scale = ['price_change', 'volume_change', 'moving_avg_5', 
                             'moving_avg_10', 'volatility', 'geopolitical_risk',
                             'interest_rate', 'inflation', 'unemployment_rate'] + \
                            [f'competitor_{comp}_price' for comp in competitor_prices.keys()]
        
        return self.scaler.fit_transform(data[features_to_scale])

    def train_and_predict(self, features, target):
        """Train models and make predictions."""
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        predictions = {}
        
        for model_name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            predictions[model_name] = y_pred[-1]  # Use the last prediction
        
        # Ensemble prediction (average of all models)
        predictions['ensemble'] = np.mean(list(predictions.values()))
        
        return predictions

# Streamlit App
def main():
    st.set_page_config(layout="wide")
    st.title("📈 Stock Analysis and Prediction Dashboard")

    with st.sidebar:
        symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", value="AAPL").upper()
        period = st.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y"], index=0)
        analyze_button = st.button("Analyze Stock")

    if analyze_button:
        analyzer = StockAnalyzer()
        analyzer1 = AdvancedStockAnalyzer()
        stock = yf.Ticker(symbol)
        hist_data = stock.history(period=period)
       
       
        hist_data1= calculate_technical_indicators(hist_data)
        news_items1 = analyzer1.get_news(symbol)
      

        if hist_data.empty:
            st.error("No historical data found.")
            return

        # Create tabs
        tab1, tab2, tab3, tab4, tab5,tab6 ,tab7= st.tabs(["📊 Price Analysis", "📰 News Sentiment", "🔮 Predictions", "🌍 Market Volatility", "💼 Competitor Analysis","📈Forecast","🧠AI insights"])

        with tab1:
            st.subheader("📊 Stock Price Analysis")
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${hist_data['Close'][-1]:.2f}")
            col2.metric("5-Day Moving Avg", f"${hist_data['Close'].rolling(window=5).mean().iloc[-1]:.2f}")
            col3.metric("Volatility", f"{hist_data['Close'].rolling(window=5).std().iloc[-1]:.2f}")

            fig = go.Figure(data=[go.Candlestick(
                x=hist_data.index,
                open=hist_data['Open'],
                high=hist_data['High'],
                low=hist_data['Low'],
                close=hist_data['Close']
            )])
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("📰 News Sentiment Analysis")
            news_items = analyzer.get_news(symbol, 'general')
            if news_items:
                sentiments = []
                for item in news_items:
                    sentiment_score = analyzer.analyze_sentiment(item['text'])
                    sentiments.append(sentiment_score['compound'])
                    with st.expander(f"📄 {item['title']}"):
                        st.write(f"**Published:** {item['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                        st.write(f"**Sentiment Score:** {sentiment_score['compound']:.2f}")
                        st.write(f"**Risk Score:** {sentiment_score['risk_score']:.2f}")
                        st.write(item['text'][:500] + "...")
                avg_sentiment = np.mean(sentiments)
                st.metric("Average Sentiment Score", f"{avg_sentiment:.2f}")
            else:
                st.warning("No news data available.")

        with tab3:
            st.subheader("🔮 Stock Price Predictions")
            features = analyzer.prepare_features(hist_data, symbol)
            target = hist_data['Close']
            predictions = analyzer.train_and_predict(features, target)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Ensemble Prediction", f"${predictions['ensemble']:.2f}")
            col2.metric("Random Forest", f"${predictions['RandomForest']:.2f}")
            col3.metric("XGBoost", f"${predictions['XGBoost']:.2f}")
            col4.metric("Gradient Boosting", f"${predictions['GradientBoosting']:.2f}")

        with tab4:
            st.subheader("🌍 Market Volatility Analysis")
            gr_index = analyzer.get_geopolitical_risk_index()
            st.write(f"Current Geopolitical Risk Index: {gr_index:.2f}")
            st.progress(gr_index)

            # Political News
            st.markdown("### Political News Highlights")
            pol_news = analyzer.get_news(symbol, 'political')
            if pol_news:
                for item in pol_news:
                    with st.expander(f"📰 {item['title']}"):
                        st.write(f"**Published:** {item['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                        st.write(item['text'][:500] + "...")
            else:
                st.warning("No political news found.")

        with tab5:
            st.subheader("💼 Competitor Analysis")
            competitor_prices = analyzer.get_competitor_prices(symbol)
            if competitor_prices:
                for comp, price in competitor_prices.items():
                    st.metric(f"{comp} Price", f"${price:.2f}")
            else:
                st.warning("No competitor data found.")
        with tab6:
            st.subheader("Deep Learning Forecast")

            try:
                lstm_preds = analyzer1.lstm_forecast(hist_data1)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist_data1.index, y=hist_data1['Close'], name='Historical'))
                future_dates = [hist_data1.index[-1] + timedelta(days=i) for i in range(1,6)]
                fig.add_trace(go.Scatter(x=future_dates, y=lstm_preds, name='LSTM Forecast', 
                                       line=dict(color='red', dash='dot')))
                st.plotly_chart(fig)
                
                st.markdown("**Forecast Rationale:**")
                st.markdown(f"""
                - Recent Trend: {'Bullish' if hist_data1['Close'][-5] < hist_data1['Close'][-1] else 'Bearish'}
                - Volatility Index: {hist_data1['Close'].rolling(5).std()[-1]:.2f}
                - Predicted 5-Day Change: {(lstm_preds[-1] - hist_data1['Close'][-1])/hist_data1['Close'][-1]*100:.2f}%
                """)
            except Exception as e:
                st.error(f"Forecast error: choose longer time")
        with tab7:
            st.subheader("AI-Powered Market Insights")
            print( "symbol ", symbol)
            print('hist data', hist_data1)
            print(" ".join([n['text'] for n in news_items1]))

            context = f"""
            Stock Analysis Context:
            - Symbol: {symbol}
            - Current Price: ${hist_data1['Close'][-1]:.2f}
            - 30-Day Volatility: {hist_data1['Close'].rolling(30).std()[-1]:.2f}
            - RSI: {hist_data1['RSI'][-1]:.2f}
            - MACD: {hist_data1['MACD'][-1]:.2f}
            - Recent News Sentiment: {" ".join([n['text'] for n in news_items1])}
            - 50-Day MA: {hist_data1['Close'].rolling(50).mean()[-1]:.2f}
            - 200-Day MA: {hist_data1['Close'].rolling(200).mean()[-1]:.2f}
            """
            logger.info(context)
            with st.spinner("Generating AI Insights using Llama3-70b..."):
                insights = analyzer1.generate_insights(context)
            st.markdown(f"### 📈 Llama3-70b Analysis")
            st.markdown(insights)


if __name__ == "__main__":
    main()