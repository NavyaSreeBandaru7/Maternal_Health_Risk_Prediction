import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json

# ML and AI Libraries
import sklearn
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import shap
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# NLP and GenAI
import spacy
import transformers
from transformers import pipeline, AutoTokenizer, AutoModel
import openai
from langchain import LLMChain, PromptTemplate
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Web Framework
import streamlit as st
import gradio as gr

# Utilities
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import threading

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

class ConfigManager:
    """Centralized configuration management for the application."""
    
    def __init__(self):
        self.config = {
            'model_params': {
                'random_forest': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                },
                'xgboost': {
                    'n_estimators': 150,
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                },
                'neural_network': {
                    'hidden_layers': [128, 64, 32],
                    'dropout_rate': 0.3,
                    'batch_size': 32,
                    'epochs': 100,
                    'validation_split': 0.2
                }
            },
            'risk_thresholds': {
                'low': 0.3,
                'medium': 0.7,
                'high': 0.9
            },
            'feature_importance_threshold': 0.01,
            'monitoring': {
                'alert_threshold': 0.8,
                'notification_cooldown': 3600  # 1 hour in seconds
            }
        }
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation."""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

class DataProcessor:
    """Advanced data processing and feature engineering pipeline."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
    def load_and_preprocess(self, data_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load and preprocess the maternal health dataset."""
        try:
            # Load dataset
            df = pd.read_csv(data_path)
            logging.info(f"Dataset loaded: {df.shape}")
            
            # Handle missing values with advanced imputation
            df = self._advanced_imputation(df)
            
            # Feature engineering
            df = self._feature_engineering(df)
            
            # Encode categorical variables
            df = self._encode_categorical(df)
            
            # Separate features and target
            target_col = 'RiskLevel' if 'RiskLevel' in df.columns else df.columns[-1]
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Store feature names
            self.feature_names = list(X.columns)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Encode target if categorical
            if y.dtype == 'object':
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = y.values
                
            logging.info("Data preprocessing completed successfully")
            return pd.DataFrame(X_scaled, columns=self.feature_names), y_encoded
            
        except Exception as e:
            logging.error(f"Error in data preprocessing: {str(e)}")
            raise
    
    def _advanced_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform advanced missing value imputation."""
        from sklearn.impute import KNNImputer
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # KNN imputation for numeric columns
        if len(numeric_cols) > 0:
            knn_imputer = KNNImputer(n_neighbors=5)
            df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])
        
        # Mode imputation for categorical columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
                
        return df
    
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced engineered features."""
        # BMI calculation if height and weight available
        if 'Height' in df.columns and 'Weight' in df.columns:
            df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
            df['BMI_Category'] = pd.cut(df['BMI'], 
                                     bins=[0, 18.5, 25, 30, float('inf')],
                                     labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        
        # Age risk categories
        if 'Age' in df.columns:
            df['Age_Risk'] = pd.cut(df['Age'],
                                  bins=[0, 18, 35, 45, float('inf')],
                                  labels=['Teen', 'Normal', 'Advanced', 'High_Risk'])
        
        # Blood pressure categories
        if 'SystolicBP' in df.columns and 'DiastolicBP' in df.columns:
            df['BP_Category'] = 'Normal'
            df.loc[(df['SystolicBP'] >= 140) | (df['DiastolicBP'] >= 90), 'BP_Category'] = 'High'
            df.loc[(df['SystolicBP'] < 90) | (df['DiastolicBP'] < 60), 'BP_Category'] = 'Low'
        
        # Glucose level categories
        if 'BS' in df.columns:  # Blood Sugar
            df['Glucose_Category'] = pd.cut(df['BS'],
                                          bins=[0, 140, 200, float('inf')],
                                          labels=['Normal', 'Prediabetic', 'Diabetic'])
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables using multiple techniques."""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col == 'RiskLevel':  # Skip target column
                continue
                
            # Use one-hot encoding for low cardinality
            if df[col].nunique() <= 5:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df.drop(columns=[col], inplace=True)
            else:
                # Use label encoding for high cardinality
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        
        return df

class EnsemblePredictor:
    """Advanced ensemble model with multiple algorithms and explainability."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.models = {}
        self.ensemble = None
        self.explainer = None
        self.feature_names = []
        
    def build_models(self, X: pd.DataFrame, y: np.ndarray):
        """Build and train ensemble of models."""
        self.feature_names = list(X.columns)
        
        # Random Forest
        rf_params = self.config.get('model_params.random_forest')
        self.models['random_forest'] = RandomForestClassifier(**rf_params)
        
        # XGBoost
        xgb_params = self.config.get('model_params.xgboost')
        self.models['xgboost'] = xgb.XGBClassifier(**xgb_params)
        
        # Neural Network
        self.models['neural_network'] = self._build_neural_network(X.shape[1], len(np.unique(y)))
        
        # Train individual models
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                            random_state=42, stratify=y)
        
        # Train traditional ML models
        self.models['random_forest'].fit(X_train, y_train)
        self.models['xgboost'].fit(X_train, y_train)
        
        # Train neural network
        nn_params = self.config.get('model_params.neural_network')
        history = self.models['neural_network'].fit(
            X_train, tf.keras.utils.to_categorical(y_train),
            epochs=nn_params['epochs'],
            batch_size=nn_params['batch_size'],
            validation_split=nn_params['validation_split'],
            verbose=0
        )
        
        # Create voting ensemble
        self.ensemble = VotingClassifier(
            estimators=[
                ('rf', self.models['random_forest']),
                ('xgb', self.models['xgboost'])
            ],
            voting='soft'
        )
        self.ensemble.fit(X_train, y_train)
        
        # Initialize SHAP explainer
        self.explainer = shap.Explainer(self.models['random_forest'])
        
        # Evaluate models
        self._evaluate_models(X_test, y_test)
        
        logging.info("Ensemble model training completed")
        
    def _build_neural_network(self, input_dim: int, num_classes: int):
        """Build neural network architecture."""
        nn_params = self.config.get('model_params.neural_network')
        
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.BatchNormalization(),
        ])
        
        # Hidden layers
        for units in nn_params['hidden_layers']:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(nn_params['dropout_rate']))
            model.add(layers.BatchNormalization())
        
        # Output layer
        model.add(layers.Dense(num_classes, activation='softmax'))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _evaluate_models(self, X_test: pd.DataFrame, y_test: np.ndarray):
        """Evaluate all models and print performance metrics."""
        results = {}
        
        # Evaluate traditional ML models
        for name, model in self.models.items():
            if name != 'neural_network':
                y_pred = model.predict(X_test)
                accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
                results[name] = accuracy
                
                print(f"\n{name.upper()} Performance:")
                print(f"Accuracy: {accuracy:.4f}")
                print("Classification Report:")
                print(classification_report(y_test, y_pred))
        
        # Evaluate neural network
        y_pred_nn = self.models['neural_network'].predict(X_test)
        y_pred_nn_classes = np.argmax(y_pred_nn, axis=1)
        nn_accuracy = sklearn.metrics.accuracy_score(y_test, y_pred_nn_classes)
        results['neural_network'] = nn_accuracy
        
        print(f"\nNEURAL NETWORK Performance:")
        print(f"Accuracy: {nn_accuracy:.4f}")
        
        # Evaluate ensemble
        y_pred_ensemble = self.ensemble.predict(X_test)
        ensemble_accuracy = sklearn.metrics.accuracy_score(y_test, y_pred_ensemble)
        results['ensemble'] = ensemble_accuracy
        
        print(f"\nENSEMBLE Performance:")
        print(f"Accuracy: {ensemble_accuracy:.4f}")
        
        return results
    
    def predict_with_explanation(self, X: np.ndarray) -> Dict:
        """Make prediction with SHAP explanations."""
        # Get ensemble prediction
        ensemble_pred = self.ensemble.predict_proba(X)[0]
        predicted_class = np.argmax(ensemble_pred)
        confidence = np.max(ensemble_pred)
        
        # Get individual model predictions
        individual_preds = {}
        for name, model in self.models.items():
            if name != 'neural_network':
                pred = model.predict_proba(X)[0]
                individual_preds[name] = {
                    'probabilities': pred.tolist(),
                    'predicted_class': np.argmax(pred),
                    'confidence': np.max(pred)
                }
        
        # Neural network prediction
        nn_pred = self.models['neural_network'].predict(X)[0]
        individual_preds['neural_network'] = {
            'probabilities': nn_pred.tolist(),
            'predicted_class': np.argmax(nn_pred),
            'confidence': np.max(nn_pred)
        }
        
        # SHAP explanations
        shap_values = self.explainer(X)
        feature_importance = dict(zip(self.feature_names, shap_values.values[0]))
        
        # Risk level interpretation
        risk_levels = ['Low Risk', 'Medium Risk', 'High Risk']
        risk_level = risk_levels[predicted_class] if predicted_class < len(risk_levels) else 'Unknown'
        
        return {
            'prediction': {
                'risk_level': risk_level,
                'predicted_class': int(predicted_class),
                'confidence': float(confidence),
                'probabilities': ensemble_pred.tolist()
            },
            'individual_models': individual_preds,
            'explanations': {
                'feature_importance': feature_importance,
                'top_risk_factors': sorted(feature_importance.items(), 
                                         key=lambda x: abs(x[1]), reverse=True)[:5]
            },
            'recommendations': self._generate_recommendations(predicted_class, feature_importance)
        }
    
    def _generate_recommendations(self, risk_class: int, feature_importance: Dict) -> List[str]:
        """Generate personalized health recommendations."""
        recommendations = []
        
        # Get top contributing features
        top_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        # Risk-specific recommendations
        if risk_class == 2:  # High risk
            recommendations.extend([
                "⚠️ Immediate medical consultation recommended",
                "📊 Schedule comprehensive health screening",
                "🏥 Consider specialist referral"
            ])
        elif risk_class == 1:  # Medium risk
            recommendations.extend([
                "📅 Regular monitoring and follow-up required",
                "💊 Review current medications with healthcare provider",
                "🏃‍♀️ Lifestyle modifications may be beneficial"
            ])
        else:  # Low risk
            recommendations.extend([
                "✅ Continue current health management",
                "📋 Maintain regular check-ups",
                "💪 Focus on preventive care"
            ])
        
        # Feature-specific recommendations
        for feature, importance in top_features:
            if 'BP' in feature.upper():
                recommendations.append("🩺 Monitor blood pressure regularly")
            elif 'BMI' in feature.upper() or 'Weight' in feature:
                recommendations.append("⚖️ Consider weight management consultation")
            elif 'Age' in feature:
                recommendations.append("👩‍⚕️ Age-appropriate screening protocols")
            elif 'Glucose' in feature or 'BS' in feature:
                recommendations.append("🍎 Glucose monitoring and dietary consultation")
        
        return list(set(recommendations))  # Remove duplicates

class NLPAnalyzer:
    """Natural Language Processing for clinical notes and patient communication."""
    
    def __init__(self):
        self.nlp = None
        self.sentiment_analyzer = None
        self.medical_ner = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models."""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            
            # Medical NER pipeline
            self.medical_ner = pipeline("ner", 
                                      model="d4data/biomedical-ner-all",
                                      aggregation_strategy="simple")
            
            # Sentiment analysis
            self.sentiment_analyzer = pipeline("sentiment-analysis",
                                             model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            
            logging.info("NLP models initialized successfully")
            
        except Exception as e:
            logging.warning(f"Could not initialize all NLP models: {str(e)}")
            # Fallback to basic models
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.sentiment_analyzer = pipeline("sentiment-analysis")
            except:
                logging.error("Failed to initialize basic NLP models")
    
    def analyze_clinical_notes(self, text: str) -> Dict:
        """Analyze clinical notes for risk factors and insights."""
        if not text or not self.nlp:
            return {"error": "No text provided or NLP model not available"}
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract entities
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Medical entity recognition
        medical_entities = []
        if self.medical_ner:
            try:
                med_entities = self.medical_ner(text)
                medical_entities = [(entity['word'], entity['entity_group']) 
                                  for entity in med_entities]
            except Exception as e:
                logging.warning(f"Medical NER failed: {str(e)}")
        
        # Sentiment analysis
        sentiment = {"label": "NEUTRAL", "score": 0.5}
        if self.sentiment_analyzer:
            try:
                sentiment = self.sentiment_analyzer(text)[0]
            except Exception as e:
                logging.warning(f"Sentiment analysis failed: {str(e)}")
        
        # Risk keyword detection
        risk_keywords = [
            'hypertension', 'diabetes', 'preeclampsia', 'gestational diabetes',
            'high blood pressure', 'bleeding', 'pain', 'swelling', 'headache',
            'nausea', 'vomiting', 'fever', 'infection'
        ]
        
        detected_risks = [keyword for keyword in risk_keywords 
                         if keyword.lower() in text.lower()]
        
        # Extract numerical values (vitals)
        numbers = [token.text for token in doc if token.like_num]
        
        return {
            'entities': entities,
            'medical_entities': medical_entities,
            'sentiment': sentiment,
            'detected_risks': detected_risks,
            'numerical_values': numbers,
            'risk_score': len(detected_risks) / len(risk_keywords),
            'text_length': len(text.split()),
            'urgency_indicators': self._detect_urgency(text)
        }
    
    def _detect_urgency(self, text: str) -> List[str]:
        """Detect urgency indicators in clinical text."""
        urgency_patterns = [
            'urgent', 'emergency', 'immediately', 'asap', 'critical',
            'severe', 'acute', 'sudden', 'rapid', 'significant'
        ]
        
        return [pattern for pattern in urgency_patterns 
                if pattern.lower() in text.lower()]

class HealthcareAgent:
    """Conversational AI agent for healthcare providers."""
    
    def __init__(self, predictor: EnsemblePredictor, nlp_analyzer: NLPAnalyzer):
        self.predictor = predictor
        self.nlp_analyzer = nlp_analyzer
        self.memory = ConversationBufferWindowMemory(k=10)
        self.conversation_history = []
        
    def process_query(self, query: str, patient_data: Optional[Dict] = None) -> str:
        """Process natural language queries about maternal health."""
        
        # Analyze the query
        query_analysis = self.nlp_analyzer.analyze_clinical_notes(query)
        
        # Determine query intent
        intent = self._classify_intent(query, query_analysis)
        
        # Generate response based on intent
        if intent == 'risk_assessment' and patient_data:
            return self._handle_risk_assessment(query, patient_data, query_analysis)
        elif intent == 'explanation':
            return self._handle_explanation_request(query)
        elif intent == 'recommendations':
            return self._handle_recommendation_request(query, patient_data)
        elif intent == 'general_info':
            return self._handle_general_info(query)
        else:
            return self._handle_general_conversation(query)
    
    def _classify_intent(self, query: str, analysis: Dict) -> str:
        """Classify the intent of the user query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['predict', 'risk', 'assess', 'probability']):
            return 'risk_assessment'
        elif any(word in query_lower for word in ['why', 'explain', 'how', 'reason']):
            return 'explanation'
        elif any(word in query_lower for word in ['recommend', 'suggest', 'advice', 'what should']):
            return 'recommendations'
        elif any(word in query_lower for word in ['what is', 'define', 'tell me about']):
            return 'general_info'
        else:
            return 'general_conversation'
    
    def _handle_risk_assessment(self, query: str, patient_data: Dict, analysis: Dict) -> str:
        """Handle risk assessment requests."""
        try:
            # Convert patient data to model input format
            X = np.array(list(patient_data.values())).reshape(1, -1)
            
            # Get prediction
            result = self.predictor.predict_with_explanation(X)
            
            response = f"""
Based on the patient data analysis:

🎯 **Risk Assessment**: {result['prediction']['risk_level']}
📊 **Confidence**: {result['prediction']['confidence']:.1%}

🔍 **Key Risk Factors**:
"""
            for factor, importance in result['explanations']['top_risk_factors'][:3]:
                impact = "increases" if importance > 0 else "decreases"
                response += f"• {factor}: {impact} risk (impact: {abs(importance):.3f})\n"
            
            response += "\n💡 **Recommendations**:\n"
            for rec in result['recommendations'][:3]:
                response += f"• {rec}\n"
                
            # Add urgency if detected in query
            if analysis['urgency_indicators']:
                response += f"\n⚠️ **Urgency Noted**: {', '.join(analysis['urgency_indicators'])}"
            
            return response
            
        except Exception as e:
            return f"I apologize, but I encountered an error processing the risk assessment: {str(e)}"
    
    def _handle_explanation_request(self, query: str) -> str:
        """Handle requests for explanations."""
        explanations = {
            'gestational diabetes': """
Gestational diabetes occurs when blood sugar levels become elevated during pregnancy. 
Risk factors include family history, obesity, and age over 25. Regular monitoring 
and dietary management are crucial for both maternal and fetal health.
            """,
            'preeclampsia': """
Preeclampsia is characterized by high blood pressure and protein in urine after 
20 weeks of pregnancy. It can lead to serious complications if untreated. 
Warning signs include severe headaches, vision changes, and upper abdominal pain.
            """,
            'maternal mortality': """
Maternal mortality refers to deaths during pregnancy, childbirth, or within 42 days 
of delivery. Leading causes include hemorrhage, hypertensive disorders, and sepsis. 
Early detection and intervention are key to prevention.
            """
        }
        
        query_lower = query.lower()
        for topic, explanation in explanations.items():
            if topic in query_lower:
                return f"📚 **{topic.title()}**:\n{explanation.strip()}"
        
        return "I'd be happy to explain maternal health topics. Could you be more specific about what you'd like to know?"
    
    def _handle_recommendation_request(self, query: str, patient_data: Optional[Dict]) -> str:
        """Handle recommendation requests."""
        if not patient_data:
            return """
For personalized recommendations, I would need patient data. However, here are general maternal health recommendations:

🏥 **Prenatal Care**:
• Regular check-ups with healthcare provider
• Proper nutrition and prenatal vitamins
• Avoid alcohol, smoking, and harmful substances

📊 **Monitoring**:
• Track blood pressure and weight
• Monitor fetal movements
• Watch for warning signs

💪 **Lifestyle**:
• Moderate exercise as approved by doctor
• Adequate rest and stress management
• Stay hydrated and maintain healthy diet
            """
        
        # Generate personalized recommendations based on data
        return "Based on the patient profile, here are personalized recommendations..."
    
    def _handle_general_info(self, query: str) -> str:
        """Handle general information requests."""
        info_topics = {
            'maternal health': "Maternal health encompasses the health of women during pregnancy, childbirth, and postpartum period.",
            'prenatal care': "Prenatal care involves regular medical check-ups during pregnancy to monitor health and prevent complications.",
            'risk factors': "Common risk factors include advanced maternal age, multiple pregnancies, chronic conditions, and lifestyle factors."
        }
        
        query_lower = query.lower()
        for topic, info in info_topics.items():
            if topic in query_lower:
                return f"ℹ️ **{topic.title()}**: {info}"
        
        return "I can provide information about maternal health topics. What would you like to know more about?"
    
    def _handle_general_conversation(self, query: str) -> str:
        """Handle general conversation."""
        return """
I'm here to help with maternal health risk assessment and provide healthcare insights. 
I can assist with:

• 🔍 Risk predictions and assessments
• 📊 Data analysis and explanations  
• 💡 Clinical recommendations
• 📚 Educational information
• 🤖 Answer questions about maternal health

How can I help you today?
        """

class RealTimeMonitor:
    """Real-time monitoring and alerting system."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.active_monitors = {}
        self.alert_history = []
        self.running = False
        
    def start_monitoring(self, patient_id: str, data_stream: callable):
        """Start monitoring a patient's vital signs."""
        if patient_id not in self.active_monitors:
            self.active_monitors[patient_id] = {
                'data_stream': data_stream,
                'last_alert': None,
                'status': 'active'
            }
            
        threading.Thread(target=self._monitor_patient, 
                        args=(patient_id,), daemon=True).start()
        
        logging.info(f"Started monitoring patient {patient_id}")
    
    def _monitor_patient(self, patient_id: str):
        """Monitor individual patient in background thread."""
        monitor = self.active_monitors[patient_id]
        alert_threshold = self.config.get('monitoring.alert_threshold')
        cooldown = self.config.get('monitoring.notification_cooldown')
        
        while monitor['status'] == 'active':
            try:
                # Get latest data
                current_data = monitor['data_stream']()
                
                if current_data is not None:
                    # Simulate risk assessment
                    risk_score = self._calculate_risk_score(current_data)
                    
                    # Check for alerts
                    if risk_score > alert_threshold:
                        now = datetime.now()
                        last_alert = monitor.get('last_alert')
                        
                        # Check cooldown period
                        if (last_alert is None or 
                            (now - last_alert).seconds > cooldown):
                            
                            self._trigger_alert(patient_id, risk_score, current_data)
                            monitor['last_alert'] = now
                
                # Sleep before next check
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logging.error(f"Monitoring error for patient {patient_id}: {str(e)}")
                time.sleep(60)  # Wait longer on error
    
    def _calculate_risk_score(self, data: Dict) -> float:
        """Calculate risk score from current data."""
        # Simplified risk scoring - in production, use trained model
        risk_factors = 0
        total_factors = 0
        
        # Blood pressure check
        if 'systolic_bp' in data:
            total_factors += 1
            if data['systolic_bp'] > 140 or data.get('diastolic_bp', 0) > 90:
                risk_factors += 1
        
        # Heart rate check
        if 'heart_rate' in data:
            total_factors += 1
            if data['heart_rate'] > 100 or data['heart_rate'] < 60:
                risk_factors += 1
        
        # Blood sugar check
        if 'blood_sugar' in data:
            total_factors += 1
            if data['blood_sugar'] > 140:
                risk_factors += 1
        
        # Temperature check
        if 'temperature' in data:
            total_factors += 1
            if data['temperature'] > 100.4 or data['temperature'] < 96:
                risk_factors += 1
        
        return risk_factors / max(total_factors, 1)
    
    def _trigger_alert(self, patient_id: str, risk_score: float, data: Dict):
        """Trigger alert for high-risk patient."""
        alert = {
            'patient_id': patient_id,
            'timestamp': datetime.now(),
            'risk_score': risk_score,
            'data': data,
            'alert_type': 'high_risk',
            'status': 'active'
        }
        
        self.alert_history.append(alert)
        
        # Log alert
        logging.warning(f"HIGH RISK ALERT - Patient {patient_id}: Risk Score {risk_score:.2f}")
        
        # In production, this would send notifications via email, SMS, etc.
        print(f"🚨 ALERT: Patient {patient_id} shows elevated risk (Score: {risk_score:.2f})")
        
    def stop_monitoring(self, patient_id: str):
        """Stop monitoring a patient."""
        if patient_id in self.active_monitors:
            self.active_monitors[patient_id]['status'] = 'stopped'
            logging.info(f"Stopped monitoring patient {patient_id}")
    
    def get_alert_summary(self) -> Dict:
        """Get summary of recent alerts."""
        recent_alerts = [alert for alert in self.alert_history 
                        if (datetime.now() - alert['timestamp']).days < 7]
        
        return {
            'total_alerts': len(self.alert_history),
            'recent_alerts': len(recent_alerts),
            'active_monitors': len([m for m in self.active_monitors.values() 
                                  if m['status'] == 'active']),
            'latest_alerts': recent_alerts[-5:] if recent_alerts else []
        }

class VisualizationEngine:
    """Advanced data visualization and dashboard generation."""
    
    def __init__(self):
        self.color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
    def create_risk_dashboard(self, predictions: List[Dict], patient_data: pd.DataFrame) -> Dict:
        """Create comprehensive risk assessment dashboard."""
        
        # Risk distribution chart
        risk_dist_fig = self._create_risk_distribution(predictions)
        
        # Feature importance chart
        feature_imp_fig = self._create_feature_importance(predictions[0] if predictions else None)
        
        # Patient timeline
        timeline_fig = self._create_patient_timeline(patient_data)
        
        # Risk correlation heatmap
        correlation_fig = self._create_correlation_heatmap(patient_data)
        
        return {
            'risk_distribution': risk_dist_fig,
            'feature_importance': feature_imp_fig,
            'patient_timeline': timeline_fig,
            'correlation_heatmap': correlation_fig
        }
    
    def _create_risk_distribution(self, predictions: List[Dict]) -> go.Figure:
        """Create risk level distribution chart."""
        if not predictions:
            return go.Figure()
        
        risk_levels = [pred['prediction']['risk_level'] for pred in predictions]
        risk_counts = pd.Series(risk_levels).value_counts()
        
        fig = go.Figure(data=[
            go.Pie(labels=risk_counts.index, 
                   values=risk_counts.values,
                   hole=0.4,
                   marker_colors=self.color_palette)
        ])
        
        fig.update_layout(
            title="Risk Level Distribution",
            font=dict(size=14),
            showlegend=True
        )
        
        return fig
    
    def _create_feature_importance(self, prediction: Dict) -> go.Figure:
        """Create feature importance chart."""
        if not prediction or 'explanations' not in prediction:
            return go.Figure()
        
        top_features = prediction['explanations']['top_risk_factors'][:10]
        features, importance = zip(*top_features)
        
        colors = ['red' if imp > 0 else 'blue' for _, imp in top_features]
        
        fig = go.Figure(data=[
            go.Bar(x=list(importance), 
                   y=list(features),
                   orientation='h',
                   marker_color=colors)
        ])
        
        fig.update_layout(
            title="Top Risk Factors",
            xaxis_title="SHAP Value",
            yaxis_title="Features",
            height=400
        )
        
        return fig
    
    def _create_patient_timeline(self, patient_data: pd.DataFrame) -> go.Figure:
        """Create patient monitoring timeline."""
        if patient_data.empty:
            return go.Figure()
        
        # Simulate timeline data
        dates = pd.date_range(start='2024-01-01', periods=len(patient_data), freq='D')
        
        fig = go.Figure()
        
        # Add multiple vital signs if available
        if 'SystolicBP' in patient_data.columns:
            fig.add_trace(go.Scatter(
                x=dates, 
                y=patient_data['SystolicBP'],
                mode='lines+markers',
                name='Systolic BP',
                line=dict(color='red')
            ))
        
        if 'BS' in patient_data.columns:
            fig.add_trace(go.Scatter(
                x=dates, 
                y=patient_data['BS'],
                mode='lines+markers',
                name='Blood Sugar',
                line=dict(color='blue'),
                yaxis='y2'
            ))
        
        fig.update_layout(
            title="Patient Vital Signs Timeline",
            xaxis_title="Date",
            yaxis=dict(title="Blood Pressure", side="left", color="red"),
            yaxis2=dict(title="Blood Sugar", side="right", overlaying="y", color="blue"),
            height=400
        )
        
        return fig
    
    def _create_correlation_heatmap(self, patient_data: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap of features."""
        if patient_data.empty:
            return go.Figure()
        
        # Select numeric columns
        numeric_cols = patient_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return go.Figure()
        
        correlation_matrix = patient_data[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            height=500
        )
        
        return fig

class WebInterface:
    """Streamlit-based web interface for the application."""
    
    def __init__(self, predictor: EnsemblePredictor, agent: HealthcareAgent, 
                 nlp_analyzer: NLPAnalyzer, visualizer: VisualizationEngine):
        self.predictor = predictor
        self.agent = agent
        self.nlp_analyzer = nlp_analyzer
        self.visualizer = visualizer
        
    def run_streamlit_app(self):
        """Run the Streamlit web application."""
        st.set_page_config(
            page_title="Maternal Health Risk Prediction System",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main {
            padding-top: 2rem;
        }
        .stAlert {
            margin-top: 1rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.title("🏥 Maternal Health Risk Prediction System")
        st.markdown("*AI-powered healthcare tool for maternal risk assessment*")
        
        # Sidebar
        with st.sidebar:
            st.header("Navigation")
            page = st.selectbox("Choose a page:", [
                "Risk Prediction",
                "Clinical Notes Analysis", 
                "AI Healthcare Assistant",
                "Real-time Monitoring",
                "Analytics Dashboard"
            ])
        
        # Main content based on selected page
        if page == "Risk Prediction":
            self._risk_prediction_page()
        elif page == "Clinical Notes Analysis":
            self._clinical_notes_page()
        elif page == "AI Healthcare Assistant":
            self._ai_assistant_page()
        elif page == "Real-time Monitoring":
            self._monitoring_page()
        else:
            self._dashboard_page()
    
    def _risk_prediction_page(self):
        """Risk prediction interface."""
        st.header("🎯 Maternal Health Risk Prediction")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Patient Information")
            
            # Input fields
            age = st.slider("Age", 15, 50, 25)
            systolic_bp = st.slider("Systolic Blood Pressure", 80, 200, 120)
            diastolic_bp = st.slider("Diastolic Blood Pressure", 40, 120, 80)
            blood_sugar = st.slider("Blood Sugar Level", 60, 300, 100)
            body_temp = st.slider("Body Temperature (°F)", 96, 104, 98.6)
            heart_rate = st.slider("Heart Rate", 50, 150, 70)
            
            # Additional clinical info
            st.subheader("Additional Information")
            previous_complications = st.checkbox("Previous pregnancy complications")
            family_history = st.checkbox("Family history of diabetes/hypertension")
            multiple_pregnancy = st.checkbox("Multiple pregnancy (twins, etc.)")
            
        with col2:
            st.subheader("Risk Assessment")
            
            if st.button("Predict Risk", type="primary"):
                # Prepare input data
                input_data = np.array([
                    age, systolic_bp, diastolic_bp, blood_sugar, 
                    body_temp, heart_rate, int(previous_complications),
                    int(family_history), int(multiple_pregnancy)
                ]).reshape(1, -1)
                
                # Make prediction
                try:
                    result = self.predictor.predict_with_explanation(input_data)
                    
                    # Display results
                    risk_level = result['prediction']['risk_level']
                    confidence = result['prediction']['confidence']
                    
                    # Risk level alert
                    if risk_level == "High Risk":
                        st.error(f"⚠️ {risk_level} (Confidence: {confidence:.1%})")
                    elif risk_level == "Medium Risk":
                        st.warning(f"⚡ {risk_level} (Confidence: {confidence:.1%})")
                    else:
                        st.success(f"✅ {risk_level} (Confidence: {confidence:.1%})")
                    
                    # Feature importance
                    st.subheader("Key Risk Factors")
                    for factor, importance in result['explanations']['top_risk_factors'][:5]:
                        impact = "↗️" if importance > 0 else "↘️"
                        st.write(f"{impact} **{factor}**: {abs(importance):.3f}")
                    
                    # Recommendations
                    st.subheader("Recommendations")
                    for rec in result['recommendations']:
                        st.write(f"• {rec}")
                        
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
    
    def _clinical_notes_page(self):
        """Clinical notes analysis interface."""
        st.header("📝 Clinical Notes Analysis")
        
        st.markdown("Analyze clinical notes using advanced NLP to extract insights and risk indicators.")
        
        # Text input
        clinical_notes = st.text_area(
            "Enter clinical notes:", 
            height=200,
            placeholder="Patient presents with elevated blood pressure during routine check-up..."
        )
        
        if st.button("Analyze Notes", type="primary") and clinical_notes:
            with st.spinner("Analyzing clinical notes..."):
                analysis = self.nlp_analyzer.analyze_clinical_notes(clinical_notes)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📊 Analysis Results")
                    
                    # Risk score
                    risk_score = analysis.get('risk_score', 0)
                    st.metric("Risk Score", f"{risk_score:.2%}")
                    
                    # Sentiment
                    sentiment = analysis.get('sentiment', {})
                    st.metric("Sentiment", sentiment.get('label', 'Unknown'))
                    
                    # Detected risks
                    if analysis.get('detected_risks'):
                        st.subheader("🚨 Detected Risk Factors")
                        for risk in analysis['detected_risks']:
                            st.write(f"• {risk}")
                    
                    # Urgency indicators
                    if analysis.get('urgency_indicators'):
                        st.subheader("⚠️ Urgency Indicators")
                        for indicator in analysis['urgency_indicators']:
                            st.write(f"• {indicator}")
                
                with col2:
                    st.subheader("🔍 Extracted Information")
                    
                    # Medical entities
                    if analysis.get('medical_entities'):
                        st.write("**Medical Entities:**")
                        for entity, label in analysis['medical_entities'][:10]:
                            st.write(f"• {entity} ({label})")
                    
                    # General entities
                    if analysis.get('entities'):
                        st.write("**General Entities:**")
                        for entity, label in analysis['entities'][:10]:
                            st.write(f"• {entity} ({label})")
                    
                    # Numerical values
                    if analysis.get('numerical_values'):
                        st.write("**Numerical Values:**")
                        for value in analysis['numerical_values'][:10]:
                            st.write(f"• {value}")
    
    def _ai_assistant_page(self):
        """AI healthcare assistant interface."""
        st.header("🤖 AI Healthcare Assistant")
        
        st.markdown("Chat with our AI assistant for healthcare insights and guidance.")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "Hello! I'm your AI healthcare assistant. I can help with maternal health risk assessments, answer questions, and provide clinical insights. How can I assist you today?"
            })
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me about maternal health..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = self.agent.process_query(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
    
    def _monitoring_page(self):
        """Real-time monitoring interface."""
        st.header("📡 Real-time Patient Monitoring")
        
        st.markdown("Monitor patient vital signs in real-time with automated alerts.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Patient Vitals Simulator")
            
            patient_id = st.text_input("Patient ID", "PATIENT_001")
            
            # Simulate real-time data
            if st.button("Start Monitoring", type="primary"):
                st.success(f"Monitoring started for {patient_id}")
                
                # Simulate vitals
                vitals_data = {
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'heart_rate': np.random.normal(75, 10),
                    'systolic_bp': np.random.normal(120, 15),
                    'diastolic_bp': np.random.normal(80, 10),
                    'temperature': np.random.normal(98.6, 1),
                    'blood_sugar': np.random.normal(100, 20)
                }
                
                # Display current vitals
                st.subheader("Current Vitals")
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                
                with metrics_col1:
                    st.metric("Heart Rate", f"{vitals_data['heart_rate']:.0f} bpm")
                    st.metric("Temperature", f"{vitals_data['temperature']:.1f}°F")
                
                with metrics_col2:
                    st.metric("Systolic BP", f"{vitals_data['systolic_bp']:.0f} mmHg")
                    st.metric("Diastolic BP", f"{vitals_data['diastolic_bp']:.0f} mmHg")
                
                with metrics_col3:
                    st.metric("Blood Sugar", f"{vitals_data['blood_sugar']:.0f} mg/dL")
                    
                    # Risk indicator
                    risk_score = 0.3 + np.random.random() * 0.4  # Simulate risk score
                    if risk_score > 0.7:
                        st.error(f"⚠️ High Risk: {risk_score:.2%}")
                    elif risk_score > 0.4:
                        st.warning(f"⚡ Medium Risk: {risk_score:.2%}")
                    else:
                        st.success(f"✅ Low Risk: {risk_score:.2%}")
        
        with col2:
            st.subheader("Alert History")
            
            # Simulate alert history
            alert_data = [
                {"time": "14:30", "patient": "PATIENT_001", "type": "Blood Pressure", "severity": "Medium"},
                {"time": "13:15", "patient": "PATIENT_002", "type": "Heart Rate", "severity": "High"},
                {"time": "12:45", "patient": "PATIENT_001", "type": "Temperature", "severity": "Low"},
            ]
            
            for alert in alert_data:
                severity_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                st.write(f"{severity_color[alert['severity']]} {alert['time']} - {alert['patient']}: {alert['type']} alert")
    
    def _dashboard_page(self):
        """Analytics dashboard interface."""
        st.header("📊 Analytics Dashboard")
        
        st.markdown("Comprehensive analytics and insights from the maternal health system.")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", "1,247", "+23")
        with col2:
            st.metric("High Risk Cases", "89", "+5")
        with col3:
            st.metric("Model Accuracy", "94.2%", "+1.2%")
        with col4:
            st.metric("Active Monitors", "12", "-2")
        
        # Charts
        st.subheader("Risk Distribution Over Time")
        
        # Generate sample data for visualization
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        risk_data = np.random.choice(['Low', 'Medium', 'High'], 
                                   size=len(dates), 
                                   p=[0.7, 0.25, 0.05])
        
        df_viz = pd.DataFrame({'Date': dates, 'Risk_Level': risk_data})
        risk_counts = df_viz.groupby(['Date', 'Risk_Level']).size().unstack(fill_value=0)
        
        # Plotly chart
        fig = go.Figure()
        
        for risk_level in ['Low', 'Medium', 'High']:
            if risk_level in risk_counts.columns:
                fig.add_trace(go.Scatter(
                    x=risk_counts.index,
                    y=risk_counts[risk_level].rolling(window=30).mean(),
                    mode='lines',
                    name=f'{risk_level} Risk',
                    stackgroup='one'
                ))
        
        fig.update_layout(
            title="30-Day Moving Average of Risk Assessments",
            xaxis_title="Date",
            yaxis_title="Number of Cases",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance across all predictions
        st.subheader("Global Feature Importance")
        
        # Simulate feature importance data
        features = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
        importance = np.random.random(len(features))
        
        fig_importance = go.Figure(data=[
            go.Bar(x=features, y=importance, marker_color=self.visualizer.color_palette)
        ])
        
        fig_importance.update_layout(
            title="Average Feature Importance Across All Predictions",
            xaxis_title="Features",
            yaxis_title="Importance Score",
            height=400
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)

def main():
    """Main application entry point."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('maternal_health_system.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Maternal Health Risk Prediction System")
    
    try:
        # Initialize components
        config_manager = ConfigManager()
        data_processor = DataProcessor(config_manager)
        predictor = EnsemblePredictor(config_manager)
        nlp_analyzer = NLPAnalyzer()
        visualizer = VisualizationEngine()
        monitor = RealTimeMonitor(config_manager)
        
        # Load and process data
        logger.info("Loading and processing data...")
        
        # For demo purposes, create sample data if no dataset is available
        sample_data = pd.DataFrame({
            'Age': np.random.normal(28, 6, 1000),
            'SystolicBP': np.random.normal(120, 20, 1000),
            'DiastolicBP': np.random.normal(80, 15, 1000),
            'BS': np.random.normal(100, 30, 1000),
            'BodyTemp': np.random.normal(98.6, 1.5, 1000),
            'HeartRate': np.random.normal(75, 15, 1000),
            'RiskLevel': np.random.choice([0, 1, 2], 1000, p=[0.7, 0.25, 0.05])
        })
        
        # Process data
        X, y = data_processor.load_and_preprocess_sample(sample_data)
        
        # Train models
        logger.info("Training ensemble models...")
        predictor.build_models(X, y)
        
        # Initialize healthcare agent
        agent = HealthcareAgent(predictor, nlp_analyzer)
        
        # Create web interface
        web_interface = WebInterface(predictor, agent, nlp_analyzer, visualizer)
        
        # Run Streamlit app
        web_interface.run_streamlit_app()
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
