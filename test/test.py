
from venv import logger
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn.functional as F
import tensorflow as tf
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime, timedelta
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

import requests

# Configure API keys (Replace with your actual keys)
FRED_API_KEY = '5a24dcab75c5f77ff277b689babdc8da'
OPENAI_API_KEY ='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'  # For GPT-4 insights
from langchain_community.llms import Ollama
ollama = Ollama(
    base_url="http://jo3m4y06rnnwhaz.askbhunte.com",
    model="llama3.2:latest"
)

class AdvancedStockAnalyzer:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        self.finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        self.openrouter_key = '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' 
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        
    # def get_news(self, symbol):
    #     """Fetch news from Yahoo Finance"""
    #     try:
    #         stock = yf.Ticker(symbol)
    #         news = stock.news
    #         return [{
    #             'title': item['title'],
    #             'text': item.get('summary', ''),
    #             'timestamp': datetime.fromtimestamp(item['providerPublishTime'])
    #         } for item in news][:10]  # Limit to 10 articles
    #     except Exception as e:
    #         st.error(f"News Error: {str(e)}")
    #         return []

    def create_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mean_squared_error')
        return model

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
            # response = requests.post(self.openrouter_url, headers=headers, json=data)
            a =ollama.invoke(prompt)
            if a !="":
                print(a)
                return a
            return "Error: Failed to get valid response from AI model"
        except Exception as e:
            return f"Connection Error: {str(e)}"

    def advanced_sentiment_analysis(self, text):
     """Perform financial sentiment analysis with proper gradient handling"""
     try:
        # Tokenize input text
        inputs = self.finbert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Inference without gradient tracking
        with torch.no_grad():
            outputs = self.finbert_model(**inputs)
        
        # Convert to probabilities
        probs = F.softmax(outputs.logits, dim=-1)
        
        # Move to CPU and convert to numpy
        probs_np = probs.detach().cpu().numpy().flatten()
        
        return {
            'sentiment': ['positive', 'neutral', 'negative'][probs_np.argmax()],
            'confidence': probs_np.max().item()
        }
     except Exception as e:
        print(f"Sentiment analysis error: {str(e)}")
        return {'sentiment': 'neutral', 'confidence': 0.0}
    def get_news(self, symbol):
     """Fetch news from Yahoo Finance with enhanced error handling"""
     try:
        stock = yf.Ticker(symbol)
        news = stock.news
        news_items = []
        
        for item in news[:10]:  # Process up to 10 articles
            try:
                # Extract timestamp with validation
                pub_time = item.get('providerPublishTime', None)
                if pub_time:
                    # Convert milliseconds to seconds if necessary
                    if pub_time > 253402300800:  # Year 10000 threshold
                        pub_time = pub_time / 1000
                    timestamp = datetime.fromtimestamp(pub_time)
                else:
                    timestamp = datetime.now()

                news_items.append({
                    'title': item.get('title', 'No Title Available'),
                    'text': item.get('summary', 'No Content Available'),
                    'timestamp': timestamp
                })
            except Exception as e:
                print(f"Skipped news item due to error: {str(e)}")
                continue
                
        return news_items if news_items else []
        
     except Exception as e:
        # st.error(f"News Error: {str(e)}")
        return []

    def prepare_lstm_data(self, data, look_back=30):
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1,1))
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])
        return np.array(X), np.array(y)

    def lstm_forecast(self, data, days=5):
        look_back = 30
        X, y = self.prepare_lstm_data(data['Close'], look_back)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        model = self.create_lstm_model((look_back, 1))
        model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        
        last_sequence = data['Close'].values[-look_back:]
        predictions = []
        for _ in range(days):
            seq_scaled = self.scaler.transform(last_sequence.reshape(-1,1))
            pred = model.predict(seq_scaled.reshape(1, look_back, 1))
            predictions.append(self.scaler.inverse_transform(pred)[0][0])
            last_sequence = np.append(last_sequence[1:], pred)
        
        return predictions

def calculate_technical_indicators(data):
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    return data

