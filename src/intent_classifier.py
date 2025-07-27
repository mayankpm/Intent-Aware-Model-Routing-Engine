import re
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

from .models import IntentType, IntentClassification


class IntentClassifier:
    """Intent classification engine using rule-based and ML approaches"""
    
    def __init__(self):
        self.keywords = {
            IntentType.CODE_GENERATION: [
                'code', 'program', 'function', 'class', 'algorithm', 'debug', 'bug', 'error',
                'python', 'javascript', 'java', 'c++', 'sql', 'api', 'database', 'server',
                'compile', 'syntax', 'variable', 'loop', 'recursion', 'data structure'
            ],
            IntentType.CREATIVE_WRITING: [
                'story', 'write', 'creative', 'poem', 'novel', 'fiction', 'character',
                'plot', 'narrative', 'imagine', 'describe', 'scene', 'dialogue',
                'creative writing', 'fiction writing', 'storytelling'
            ],
            IntentType.GENERAL_REASONING: [
                'analyze', 'explain', 'why', 'how', 'reason', 'logic', 'think',
                'problem', 'solve', 'understand', 'concept', 'theory', 'principle',
                'compare', 'contrast', 'evaluate', 'assess', 'examine'
            ],
            IntentType.CUSTOM_REASONING: [
                'domain specific', 'specialized', 'expert', 'professional', 'technical',
                'industry', 'business logic', 'workflow', 'process', 'procedure',
                'custom', 'specialized reasoning', 'domain expertise'
            ],
            IntentType.CONVERSATION: [
                'chat', 'talk', 'conversation', 'hello', 'hi', 'how are you',
                'casual', 'informal', 'friendly', 'social', 'small talk'
            ],
            IntentType.SPECIALIZED_TASKS: [
                'translate', 'summarize', 'summarization', 'translation', 'paraphrase',
                'extract', 'extraction', 'classify', 'classification', 'sentiment',
                'sentiment analysis', 'named entity', 'ner', 'question answering'
            ]
        }
        
        self.ml_classifier = None
        self.vectorizer = None
        self._train_ml_classifier()
    
    def _train_ml_classifier(self):
        """Train a simple ML classifier for intent detection"""
        # This is a simplified training - in production, you'd use a proper dataset
        training_data = []
        training_labels = []
        
        # Generate synthetic training data based on keywords
        for intent, keywords in self.keywords.items():
            for keyword in keywords:
                training_data.append(f"Can you help me with {keyword}?")
                training_labels.append(intent.value)
        
        # Add some variations
        variations = [
            ("Write a function to", IntentType.CODE_GENERATION),
            ("Create a story about", IntentType.CREATIVE_WRITING),
            ("Explain the logic behind", IntentType.GENERAL_REASONING),
            ("Use specialized knowledge to", IntentType.CUSTOM_REASONING),
            ("Let's chat about", IntentType.CONVERSATION),
            ("Translate this text", IntentType.SPECIALIZED_TASKS)
        ]
        
        for text, intent in variations:
            training_data.append(text)
            training_labels.append(intent.value)
        
        # Train TF-IDF vectorizer and classifier
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = self.vectorizer.fit_transform(training_data)
        self.ml_classifier = MultinomialNB()
        self.ml_classifier.fit(X, training_labels)
    
    def classify_intent(self, query: str) -> IntentClassification:
        """Classify the intent of a query using multiple approaches"""
        query_lower = query.lower()
        
        # Rule-based classification
        rule_based_result = self._rule_based_classification(query_lower)
        
        # ML-based classification
        ml_result = self._ml_based_classification(query_lower)
        
        # Combine results
        final_intent, confidence, reasoning = self._combine_classifications(
            rule_based_result, ml_result, query_lower
        )
        
        # Check for multi-intent
        sub_intents = self._detect_multi_intent(query_lower)
        
        return IntentClassification(
            intent=final_intent,
            confidence=confidence,
            sub_intents=sub_intents if len(sub_intents) > 1 else None,
            reasoning=reasoning
        )
    
    def _rule_based_classification(self, query: str) -> Tuple[IntentType, float, str]:
        """Rule-based intent classification using keywords"""
        scores = {}
        
        for intent, keywords in self.keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in query:
                    score += 1
            
            if score > 0:
                scores[intent] = score / len(keywords)
        
        if not scores:
            return IntentType.CONVERSATION, 0.3, "No specific intent detected, defaulting to conversation"
        
        # Find the best match
        best_intent = max(scores.items(), key=lambda x: x[1])
        confidence = min(best_intent[1] * 2, 0.9)  # Scale confidence
        
        return best_intent[0], confidence, f"Rule-based classification based on keyword matching"
    
    def _ml_based_classification(self, query: str) -> Tuple[IntentType, float, str]:
        """ML-based intent classification"""
        if self.ml_classifier is None or self.vectorizer is None:
            return IntentType.CONVERSATION, 0.5, "ML classifier not available"
        
        try:
            X = self.vectorizer.transform([query])
            prediction = self.ml_classifier.predict(X)[0]
            confidence = np.max(self.ml_classifier.predict_proba(X))
            
            intent = IntentType(prediction)
            return intent, confidence, f"ML-based classification with confidence {confidence:.2f}"
        except Exception as e:
            return IntentType.CONVERSATION, 0.5, f"ML classification failed: {str(e)}"
    
    def _combine_classifications(self, rule_result: Tuple[IntentType, float, str],
                               ml_result: Tuple[IntentType, float, str],
                               query: str) -> Tuple[IntentType, float, str]:
        """Combine rule-based and ML-based classifications"""
        rule_intent, rule_conf, rule_reason = rule_result
        ml_intent, ml_conf, ml_reason = ml_result
        
        # If both agree, use higher confidence
        if rule_intent == ml_intent:
            confidence = max(rule_conf, ml_conf)
            reasoning = f"Both rule-based and ML classifiers agree: {rule_reason}"
            return rule_intent, confidence, reasoning
        
        # If they disagree, use the one with higher confidence
        if rule_conf > ml_conf:
            return rule_intent, rule_conf, f"Rule-based classification chosen: {rule_reason}"
        else:
            return ml_intent, ml_conf, f"ML-based classification chosen: {ml_reason}"
    
    def _detect_multi_intent(self, query: str) -> List[IntentType]:
        """Detect if query has multiple intents"""
        detected_intents = []
        
        for intent, keywords in self.keywords.items():
            for keyword in keywords:
                if keyword in query:
                    detected_intents.append(intent)
                    break
        
        # Remove duplicates
        return list(set(detected_intents))
    
    def should_use_custom_reasoning(self, query: str, confidence: float) -> bool:
        """Determine if custom reasoning should be used"""
        # Check for domain-specific keywords
        custom_keywords = [
            'business logic', 'workflow', 'process', 'procedure', 'domain specific',
            'specialized', 'expert', 'professional', 'technical', 'industry specific'
        ]
        
        query_lower = query.lower()
        has_custom_keywords = any(keyword in query_lower for keyword in custom_keywords)
        
        # Use custom reasoning if high confidence in custom intent or specific keywords
        return (confidence > 0.7 and has_custom_keywords) or confidence > 0.9 