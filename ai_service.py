"""
AI Service Module for Personal Finance Chatbot
Handles Google Gemini API and Hugging Face integration for NLU and response generation
"""

import os
import json
import requests
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from transformers import pipeline
import torch
import traceback

# Load environment variables
load_dotenv()

class AIService:
    """Service class for AI interactions using Gemini API and Hugging Face models"""
    
    def __init__(self):
        """Initialize AI service with Gemini and Hugging Face"""
        # Google Gemini API configuration
        self.gemini_api_key = "AIzaSyCqUFQ_PYrAkLP23CeXwykWOQxVodb4ST0"
        self.gemini_api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
        # Hugging Face token
        self.hf_token = "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"
        
        # Use lightweight, fast models
        self.nlu_model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
        self.ner_model_name = 'dslim/bert-base-NER'
        
        # Initialize models to None (lazy loading)
        self.sentiment_analyzer = None
        self.ner_pipeline = None
        self.models_loading = False
        self.models_loaded = False
        
        # Check if APIs are available
        self.gemini_available = bool(self.gemini_api_key)
        self.hf_available = bool(self.hf_token)
        
        if self.gemini_available:
            print("✓ Google Gemini API configured")
        else:
            print("⚠ Warning: Gemini API key not found")
            
        if self.hf_available:
            print("✓ Hugging Face token configured")
            print("Models will be loaded on first request")
        else:
            print("⚠ Warning: Hugging Face token not found")
    
    def _initialize_models(self):
        """Initialize Hugging Face models for NLU tasks"""
        try:
            print("Loading NLU models...")
            
            # Initialize sentiment analysis model (fast)
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=self.nlu_model_name,
                token=self.hf_token,
                device=-1  # CPU
            )
            
            # Initialize NER model
            self.ner_pipeline = pipeline(
                "ner",
                model=self.ner_model_name,
                token=self.hf_token,
                aggregation_strategy="simple",
                device=-1  # CPU
            )
            
            print("✓ All NLU models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            traceback.print_exc()
            self.hf_available = False
    
    def analyze_nlu(self, text: str) -> Dict[str, Any]:
        """
        Analyze text using Hugging Face models for NLU tasks
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with NLU analysis results
        """
        try:
            # Initialize models on first request
            if self.hf_available and not self.models_loaded and not self.models_loading:
                self.models_loading = True
                try:
                    self._initialize_models()
                    self.models_loaded = True
                except Exception as e:
                    print(f"Failed to initialize models: {e}")
                    traceback.print_exc()
                    self.hf_available = False
                finally:
                    self.models_loading = False
            
            if not self.hf_available or not self.sentiment_analyzer or not self.ner_pipeline:
                return self._fallback_nlu_analysis(text)
            
            try:
                # Sentiment analysis
                sentiment_result = self.sentiment_analyzer(text[:512])
                sentiment = {
                    'label': sentiment_result[0]['label'].lower(),
                    'score': float(sentiment_result[0]['score'])
                }
                
                # Named Entity Recognition
                entities_result = self.ner_pipeline(text[:512])
                entities = []
                for entity in entities_result:
                    entities.append({
                        'text': str(entity['word']),
                        'type': str(entity['entity_group']),
                        'score': float(entity['score'])
                    })
                
                # Keyword extraction
                keywords = []
                for entity in entities_result:
                    keywords.append({
                        'text': str(entity['word']),
                        'relevance': float(entity['score'])
                    })
                
                # Add financial keywords
                financial_keywords = [
                    'money', 'budget', 'savings', 'expenses', 'income', 'debt',
                    'investment', 'tax', 'retirement', 'emergency fund', 'credit',
                    'loan', 'mortgage', 'insurance', 'spending', 'cost', 'price'
                ]
                
                text_lower = text.lower()
                for keyword in financial_keywords:
                    if keyword in text_lower and not any(kw['text'].lower() == keyword for kw in keywords):
                        keywords.append({'text': keyword, 'relevance': 0.8})
                
                # Categories
                categories = []
                if any(word in text_lower for word in ['budget', 'expenses', 'spending']):
                    categories.append({'label': 'Budget Management', 'score': 0.9})
                if any(word in text_lower for word in ['savings', 'investment', 'retirement']):
                    categories.append({'label': 'Savings & Investment', 'score': 0.8})
                if any(word in text_lower for word in ['debt', 'loan', 'credit']):
                    categories.append({'label': 'Debt Management', 'score': 0.7})
                
                return {
                    'sentiment': sentiment,
                    'keywords': keywords[:5],
                    'entities': entities[:5],
                    'categories': categories
                }
                
            except Exception as e:
                print(f"NLU analysis error: {e}")
                traceback.print_exc()
                return self._fallback_nlu_analysis(text)
                
        except Exception as e:
            print(f"Critical error in analyze_nlu: {e}")
            traceback.print_exc()
            return self._fallback_nlu_analysis(text)
    
    def _fallback_nlu_analysis(self, text: str) -> Dict[str, Any]:
        """Fallback NLU analysis when models unavailable"""
        try:
            keywords = []
            text_lower = text.lower()
            
            financial_keywords = [
                'money', 'budget', 'savings', 'expenses', 'income', 'debt',
                'investment', 'tax', 'retirement', 'emergency fund', 'credit',
                'loan', 'mortgage', 'insurance', 'spending', 'cost', 'price'
            ]
            
            for keyword in financial_keywords:
                if keyword in text_lower:
                    keywords.append({'text': keyword, 'relevance': 0.8})
            
            # Simple sentiment
            positive_words = ['good', 'great', 'excellent', 'improve', 'better', 'saving', 'profit']
            negative_words = ['bad', 'worse', 'debt', 'loss', 'expensive', 'struggle', 'problem']
            
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                sentiment = {'label': 'positive', 'score': 0.7}
            elif negative_count > positive_count:
                sentiment = {'label': 'negative', 'score': 0.6}
            else:
                sentiment = {'label': 'neutral', 'score': 0.5}
            
            return {
                'sentiment': sentiment,
                'keywords': keywords[:5],
                'entities': [],
                'categories': []
            }
        except Exception as e:
            print(f"Error in fallback NLU: {e}")
            return {
                'sentiment': {'label': 'neutral', 'score': 0.5},
                'keywords': [],
                'entities': [],
                'categories': []
            }
    
    def generate_response(self, prompt: str, persona: str = "general") -> str:
        """
        Generate response using Google Gemini API
        
        Args:
            prompt: Input prompt for the AI model
            persona: User persona (general, student, professional)
            
        Returns:
            Generated response text
        """
        try:
            print(f"Generating response for persona: {persona}")
            print(f"Prompt length: {len(prompt)}")
            
            # Try Gemini API first
            if self.gemini_available:
                try:
                    response = self._generate_with_gemini(prompt, persona)
                    print(f"Gemini response length: {len(response)}")
                    return response
                except Exception as e:
                    print(f"Gemini API error: {e}")
                    traceback.print_exc()
                    print("Falling back to rule-based response")
            
            # Fallback to rule-based responses
            return self._enhanced_response_generation(prompt, persona)
            
        except Exception as e:
            print(f"Critical error in generate_response: {e}")
            traceback.print_exc()
            return self._get_general_advice()
    
    def _generate_with_gemini(self, prompt: str, persona: str = "general") -> str:
        """
        Generate response using Google Gemini API
        
        Args:
            prompt: Input prompt
            persona: User persona
            
        Returns:
            Generated response text
        """
        try:
            print("Starting Gemini API call...")
            
            # Create persona-specific system instruction
            system_instructions = {
                "student": "You are a financial advisor specializing in helping students manage their finances. Focus on budgeting, student loans, part-time work, and building good financial habits early. Provide practical, actionable advice for students with limited income. Keep responses clear and structured.",
                "professional": "You are a financial advisor for working professionals. Focus on retirement planning, investment strategies, tax optimization, and wealth building. Provide sophisticated advice while considering career growth and long-term financial goals. Keep responses clear and structured.",
                "general": "You are a knowledgeable and friendly financial advisor. Provide clear, practical advice on personal finance topics including budgeting, saving, investing, debt management, and financial planning. Be encouraging and actionable in your responses. Keep responses clear and structured."
            }
            
            system_instruction = system_instructions.get(persona, system_instructions["general"])
            
            # Construct full prompt
            full_prompt = f"{system_instruction}\n\nUser Question: {prompt}\n\nProvide a clear, structured financial advice response with specific recommendations and actionable steps. Format with headers using ** and bullet points."
            
            # Prepare API request
            headers = {
                "Content-Type": "application/json"
            }
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": full_prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 1024,
                    "stopSequences": []
                },
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    }
                ]
            }
            
            print(f"Making request to: {self.gemini_api_url}")
            
            # Make API request with timeout
            response = requests.post(
                f"{self.gemini_api_url}?key={self.gemini_api_key}",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            print(f"Response status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Response structure: {list(result.keys())}")
                
                # Extract generated text
                if 'candidates' in result and len(result['candidates']) > 0:
                    candidate = result['candidates'][0]
                    
                    if 'content' in candidate and 'parts' in candidate['content']:
                        parts = candidate['content']['parts']
                        if len(parts) > 0 and 'text' in parts[0]:
                            generated_text = parts[0]['text']
                            print(f"Generated text length: {len(generated_text)}")
                            
                            # Clean and validate response
                            cleaned_response = self._clean_response(generated_text)
                            
                            if self._is_valid_response(cleaned_response, prompt):
                                print("✓ Gemini response validated")
                                return cleaned_response
                            else:
                                print("⚠ Gemini response quality check failed")
                                return self._enhanced_response_generation(prompt, persona)
                
                print("⚠ Unexpected Gemini API response structure")
                print(f"Full response: {json.dumps(result, indent=2)}")
                return self._enhanced_response_generation(prompt, persona)
                
            else:
                print(f"⚠ Gemini API error: {response.status_code}")
                print(f"Response text: {response.text}")
                return self._enhanced_response_generation(prompt, persona)
                
        except requests.exceptions.Timeout:
            print("⚠ Gemini API timeout")
            return self._enhanced_response_generation(prompt, persona)
        except requests.exceptions.RequestException as e:
            print(f"⚠ Gemini API request error: {e}")
            traceback.print_exc()
            return self._enhanced_response_generation(prompt, persona)
        except Exception as e:
            print(f"⚠ Error in Gemini generation: {e}")
            traceback.print_exc()
            return self._enhanced_response_generation(prompt, persona)
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the response"""
        try:
            response = response.strip()
            
            # Remove any artifacts
            response = response.replace("**User Question:**", "").replace("**Financial Advisor:**", "")
            
            # Ensure proper formatting
            if response and not response.endswith(('.', '!', '?')):
                sentences = response.split('.')
                if len(sentences) > 1:
                    response = '.'.join(sentences[:-1]) + '.'
            
            return response.strip()
        except Exception as e:
            print(f"Error cleaning response: {e}")
            return response
    
    def _is_valid_response(self, response: str, original_prompt: str) -> bool:
        """Validate response quality"""
        try:
            if len(response.strip()) < 50:
                return False
            
            response_lower = response.lower()
            
            # Check for financial content
            financial_terms = [
                'budget', 'save', 'saving', 'money', 'financial', 'income', 'expense',
                'investment', 'debt', 'loan', 'credit', 'fund', 'account', 'plan',
                'goal', 'strategy', 'recommend', 'consider', 'advice'
            ]
            
            has_financial_content = any(term in response_lower for term in financial_terms)
            
            # Check coherence
            words = response_lower.split()
            unique_words = set(words)
            coherence_ratio = len(unique_words) / len(words) if words else 0
            
            return has_financial_content and coherence_ratio > 0.5 and len(response) > 50
        except Exception as e:
            print(f"Error validating response: {e}")
            return False
    
    def _enhanced_response_generation(self, prompt: str, persona: str = "general") -> str:
        """Enhanced rule-based response generation with persona awareness"""
        try:
            prompt_lower = prompt.lower()
            
            # Student-specific advice
            if 'student' in prompt_lower or persona == 'student':
                if any(word in prompt_lower for word in ['loan', 'debt', 'payment']):
                    return self._get_student_loan_advice(prompt_lower)
                elif any(word in prompt_lower for word in ['save', 'saving', 'money']):
                    return self._get_student_saving_advice(prompt_lower)
                elif any(word in prompt_lower for word in ['budget', 'expense']):
                    return self._get_student_budget_advice(prompt_lower)
            
            # Professional-specific advice
            elif persona == 'professional':
                if any(word in prompt_lower for word in ['investment', 'invest', 'portfolio']):
                    return self._get_professional_investment_advice(prompt_lower)
                elif any(word in prompt_lower for word in ['retirement', '401k', 'pension']):
                    return self._get_professional_retirement_advice(prompt_lower)
                elif any(word in prompt_lower for word in ['tax', 'deduction']):
                    return self._get_professional_tax_advice(prompt_lower)
            
            # General topics
            if 'budget' in prompt_lower and 'summary' in prompt_lower:
                return self._get_budget_summary()
            elif any(word in prompt_lower for word in ['grocer', 'grocery']):
                return self._get_grocery_savings_advice()
            elif 'house' in prompt_lower and ('saving' in prompt_lower or 'buy' in prompt_lower):
                return self._get_house_savings_advice()
            elif any(word in prompt_lower for word in ['save', 'saving']):
                return self._get_general_saving_advice()
            elif any(word in prompt_lower for word in ['debt', 'loan']):
                return self._get_debt_management_advice()
            elif any(word in prompt_lower for word in ['investment', 'invest']):
                return self._get_investment_advice()
            elif 'emergency' in prompt_lower:
                return self._get_emergency_fund_advice()
            
            return self._get_general_advice()
            
        except Exception as e:
            print(f"Error in enhanced response generation: {e}")
            traceback.print_exc()
            return self._get_general_advice()
    
    # Specific advice methods (keeping all original methods)
    def _get_student_loan_advice(self, prompt: str) -> str:
        return """**Student Loan Management Strategy**

Here's how to handle your student loans effectively:

**Understanding Your Loans**:
1. List all loans with balances, interest rates, servicers
2. Know the difference between federal and private loans
3. Understand your grace period and repayment options

**Repayment Strategies**:
- **Avalanche method**: Pay minimums on all, extra on highest interest rate
- **Snowball method**: Pay smallest balance first for motivation
- **Income-driven plans**: Lower payments based on income (federal loans)

**Student-Specific Tips**:
- Keep federal loans separate from private loans
- Consider auto-pay discounts (usually 0.25% rate reduction)
- Explore forgiveness programs for public service careers
- Don't ignore loans - contact servicer if having trouble

**While in School**:
- Pay interest on unsubsidized loans if possible
- Avoid borrowing more than necessary
- Look for scholarships and grants continuously

Remember: Student loans are an investment in your future earning potential."""

    def _get_student_saving_advice(self, prompt: str) -> str:
        return """**Student Saving Strategies**

Saving money as a student requires creativity and discipline:

**Smart Saving Tactics**:
1. **Round-up savings**: Round purchases up, save the difference
2. **Meal prep**: Cook in bulk, avoid dining out frequently
3. **Textbook savings**: Buy used, rent, or use library reserves
4. **Student discounts**: Always ask - many businesses offer them
5. **Free campus resources**: Use gym, events, career services

**Income Opportunities**:
- Work-study jobs on campus
- Tutoring (often pays $15-25/hour)
- Freelance skills (writing, design, coding)
- Paid internships and co-ops

**Emergency Fund for Students**:
- Start with $300-500 goal
- Keep in separate savings account
- Use only for true emergencies

Even saving $25/month as a student builds valuable habits and provides a financial cushion."""

    def _get_student_budget_advice(self, prompt: str) -> str:
        return """**Student Budgeting Guide**

Creating a budget as a student:

**Student Budget Categories**:
- **Fixed costs**: Tuition, rent, insurance, loan payments
- **Variable necessities**: Food, gas, school supplies
- **Discretionary**: Entertainment, dining out, subscriptions

**Budgeting Tips**:
1. Track income for 3 months to find average
2. Budget based on lowest month
3. Save excess from higher-income months
4. Use apps like Mint or simple spreadsheets

**Student-Specific Categories**:
- Housing: <30% of income
- Food: 15-20%
- Transportation: 10-15%
- School supplies: 5-10%
- Savings: at least 5%

Your student years are for learning financial discipline that will serve you throughout life."""

    def _get_professional_investment_advice(self, prompt: str) -> str:
        return """**Professional Investment Strategy**

As a working professional, here's how to approach investing:

**Investment Priority Order**:
1. Emergency fund: 3-6 months expenses
2. Employer 401(k) match: Free money
3. High-interest debt: Pay off credit cards
4. Max retirement accounts: 401(k), IRA
5. Taxable investment accounts

**Asset Allocation by Age**:
- 20s-30s: 80-90% stocks, 10-20% bonds
- 40s: 70-80% stocks, 20-30% bonds
- 50s+: 60-70% stocks, 30-40% bonds

**Investment Vehicles**:
- Index funds: Low fees, broad diversification
- Target-date funds: Automatic rebalancing
- ETFs: Tax-efficient, liquid
- Real estate: REITs or rental properties

Start early, invest consistently, and keep fees low."""

    def _get_professional_retirement_advice(self, prompt: str) -> str:
        return """**Professional Retirement Planning**

Maximize your retirement savings:

**Retirement Account Limits (2024)**:
- 401(k): $23,000 ($30,500 if 50+)
- IRA: $7,000 ($8,000 if 50+)

**Strategies**:
1. Get full 401(k) match
2. Automate contributions
3. Increase with raises
4. Tax diversification: Mix traditional and Roth

**Retirement Milestones**:
- Age 30: 1x annual salary saved
- Age 40: 3x annual salary
- Age 50: 6x annual salary
- Age 60: 8x annual salary

Consider working with a fee-only financial advisor."""

    def _get_professional_tax_advice(self, prompt: str) -> str:
        return """**Professional Tax Optimization**

Strategies to minimize tax burden:

**Pre-Tax Savings**:
- 401(k) contributions
- Traditional IRA (if eligible)
- HSA contributions (triple tax advantage)
- FSA for healthcare

**Tax Strategies**:
1. Maximize retirement contributions
2. Tax-loss harvesting
3. Charitable giving: Bunching donations
4. Professional development deductions

**Advanced Options**:
- Backdoor Roth IRA
- Mega backdoor Roth
- Tax-efficient fund placement

Proactive tax planning can save thousands annually."""

    def _get_budget_summary(self) -> str:
        return """**Budget Summary Analysis**

Based on financial data analysis:

**Key Insights**:
- Track your spending patterns monthly
- Identify your top 3 expense categories
- Calculate savings rate as % of income

**Recommendations**:
1. Review and categorize all expenses
2. Set up automatic savings transfers
3. Reduce discretionary spending by 10-15%
4. Build emergency fund to 3-6 months

Regular budget reviews help maintain financial health."""

    def _get_grocery_savings_advice(self) -> str:
        return """**Smart Grocery Savings Strategies**

Proven ways to reduce grocery expenses:

**Planning & Preparation**:
1. Meal planning for the week
2. Detailed shopping list
3. Check pantry before shopping
4. Never shop hungry

**Shopping Strategies**:
- Compare unit prices
- Buy store brands (save 20-30%)
- Purchase in bulk for non-perishables
- Buy seasonal produce
- Use coupons and store apps

**Smart Choices**:
- Cook at home more
- Batch cooking and freezing
- Affordable proteins: beans, eggs, chicken
- Frozen vegetables when fresh is expensive

Expected savings: 15-30% reduction in grocery bills."""

    def _get_house_savings_advice(self) -> str:
        return """**Saving for a House Purchase**

Strategic approach to home buying:

**Down Payment Planning**:
- Traditional: 10-20% of home price
- Factor in closing costs: 2-3% extra
- Set realistic timeline: 5-10 years

**Savings Strategy**:
1. Set monthly savings target
2. High-yield savings account
3. Consider investment accounts for growth
4. Automate savings transfers

**Financial Preparation**:
- Improve credit score to 750+
- Keep debt-to-income ratio below 40%
- Maintain emergency fund separately
- Research first-time buyer programs

Plan carefully and don't overstretch your finances."""

    def _get_general_saving_advice(self) -> str:
        return """**Savings and Investment Guidance**

Practical steps to improve finances:

**Immediate Actions**:
1. Emergency fund: 3-6 months expenses
2. Automate savings transfers
3. Track all spending
4. Reduce non-essential expenses

**Investment Considerations**:
- Start with employer retirement plans
- Consider Roth IRA
- Diversify investments
- Start early for compound interest

**Saving Strategies**:
- Follow 50/30/20 rule
- Use high-yield savings accounts
- Review strategy regularly

Consistency is key. Small amounts saved regularly grow significantly."""

    def _get_debt_management_advice(self) -> str:
        return """**Debt Management Strategy**

Systematic approach to managing debt:

**Debt Assessment**:
1. List all debts with interest rates
2. Calculate debt-to-income ratio
3. Identify highest interest debts

**Repayment Strategies**:
- Avalanche: Pay highest interest first
- Snowball: Pay smallest balances first
- Debt consolidation if beneficial

**Immediate Steps**:
1. Stop taking on new debt
2. Pay more than minimums
3. Negotiate lower interest rates
4. Consider balance transfer cards

Getting out of debt takes time and discipline."""

    def _get_investment_advice(self) -> str:
        return """**Investment Basics**

Starting your investment journey:

**Investment Fundamentals**:
- Start early for compound growth
- Diversify across asset classes
- Consider your risk tolerance
- Keep fees low

**Beginner Options**:
- Index funds and ETFs
- Target-date funds
- Robo-advisors
- Employer retirement plans

**Key Principles**:
- Time in market > timing market
- Regular contributions (dollar-cost averaging)
- Rebalance annually
- Stay informed but don't panic

Investing is a long-term journey."""

    def _get_emergency_fund_advice(self) -> str:
        return """**Emergency Fund Essentials**

Building your financial safety net:

**Emergency Fund Basics**:
- Purpose: Cover unexpected expenses
- Amount: 3-6 months of expenses
- Location: High-yield savings account

**Building Strategy**:
1. Start with $500-1000
2. Automate weekly/monthly transfers
3. Use windfalls (tax refunds, bonuses)
4. Temporarily direct extra income

**What Qualifies**:
✅ Job loss, medical bills, major repairs
❌ Vacations, shopping, planned expenses

**Where to Keep It**:
- High-yield savings (4-5% rates)
- Money market account
- NOT in checking or investments

Your emergency fund provides peace of mind."""

    def _get_general_advice(self) -> str:
        return """**Personal Finance Advice**

General financial principles to consider:

**Financial Foundation**:
1. Budget: Track income and expenses
2. Emergency fund: 3-6 months expenses
3. Debt management: Pay high-interest debt
4. Insurance: Protect yourself and family

**Smart Money Habits**:
- Pay yourself first (automate savings)
- Live below your means
- Avoid lifestyle inflation
- Review financial plan regularly

**Investment Basics**:
- Start early
- Diversify investments
- Consider risk tolerance
- Take advantage of employer plans

Financial success is a journey of consistent actions."""

    def get_service_status(self) -> Dict[str, bool]:
        """Get status of AI services"""
        try:
            return {
                'gemini_available': self.gemini_available,
                'huggingface_available': self.hf_available,
                'models_loaded': self.models_loaded,
                'models_loading': self.models_loading,
                'sentiment_analyzer_ready': self.sentiment_analyzer is not None,
                'ner_pipeline_ready': self.ner_pipeline is not None,
                'fallback_mode': not self.gemini_available
            }
        except Exception as e:
            print(f"Error getting service status: {e}")
            return {
                'gemini_available': False,
                'huggingface_available': False,
                'models_loaded': False,
                'models_loading': False,
                'sentiment_analyzer_ready': False,
                'ner_pipeline_ready': False,
                'fallback_mode': True
            }
"""
 Service Module for Personal Finance Chatbot
Handles Hugging Face and Gemini integration for NLU and response generation
"""
"""
AI Service Module for Personal Finance Chatbot
Handles Gemini integration for NLU and response generation
"""

# import os
# import json
# import requests
# from typing import Dict, Any, Optional, List
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# class AIService:
#     """Service class for AI interactions using Gemini models"""
    
#     def __init__(self):
#         """Initialize AI service with Gemini configuration"""
#         # Gemini configuration
#         self.gemini_api_key = "AIzaSyCqUFQ_PYrAkLP23CeXwykWOQxVodb4ST0"
#         self.gemini_api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
#         # Check if Gemini API is available
#         self.gemini_available = bool(self.gemini_api_key)
        
#         if self.gemini_available:
#             print("Gemini API configured and available")
#         else:
#             print("Warning: Gemini API key not found. Using fallback responses.")
    
#     def analyze_nlu(self, text: str) -> Dict[str, Any]:
#         """
#         Analyze text using Gemini for NLU tasks
        
#         Args:
#             text: Text to analyze
            
#         Returns:
#             Dictionary with NLU analysis results
#         """
#         if not self.gemini_available:
#             return self._fallback_nlu_analysis(text)
        
#         try:
#             # Create NLU analysis prompt for Gemini
#             nlu_prompt = f"""
#             Analyze the following text for financial sentiment, entities, and categories:
#             "{text}"
            
#             Return a JSON response with this exact structure:
#             {{
#                 "sentiment": {{
#                     "label": "positive/negative/neutral",
#                     "score": 0.95
#                 }},
#                 "keywords": [
#                     {{"text": "keyword1", "relevance": 0.9}},
#                     {{"text": "keyword2", "relevance": 0.8}}
#                 ],
#                 "entities": [
#                     {{"text": "entity1", "type": "FINANCIAL_TERM", "score": 0.9}},
#                     {{"text": "entity2", "type": "AMOUNT", "score": 0.8}}
#                 ],
#                 "categories": [
#                     {{"label": "Budget Management", "score": 0.9}},
#                     {{"label": "Savings & Investment", "score": 0.7}}
#                 ]
#             }}
            
#             Focus on financial terms, amounts, and personal finance categories.
#             """
            
#             payload = {
#                 "contents": [
#                     {
#                         "parts": [
#                             {"text": nlu_prompt}
#                         ]
#                     }
#                 ],
#                 "generationConfig": {
#                     "temperature": 0.1,
#                     "topK": 20,
#                     "topP": 0.8,
#                     "maxOutputTokens": 500,
#                 }
#             }
            
#             headers = {
#                 "Content-Type": "application/json"
#             }
            
#             url = f"{self.gemini_api_url}?key={self.gemini_api_key}"
#             response = requests.post(url, headers=headers, json=payload, timeout=30)
            
#             if response.status_code == 200:
#                 result = response.json()
#                 if 'candidates' in result and len(result['candidates']) > 0:
#                     generated_text = result['candidates'][0]['content']['parts'][0]['text']
                    
#                     # Extract JSON from response
#                     try:
#                         # Find JSON in the response
#                         start_idx = generated_text.find('{')
#                         end_idx = generated_text.rfind('}') + 1
#                         if start_idx != -1 and end_idx != -1:
#                             json_str = generated_text[start_idx:end_idx]
#                             nlu_result = json.loads(json_str)
#                             return nlu_result
#                     except json.JSONDecodeError:
#                         print("Failed to parse Gemini NLU response as JSON")
            
#             # If Gemini fails, use fallback
#             return self._fallback_nlu_analysis(text)
                
#         except Exception as e:
#             print(f"Gemini NLU analysis error: {e}")
#             return self._fallback_nlu_analysis(text)
    
#     def _fallback_nlu_analysis(self, text: str) -> Dict[str, Any]:
#         """
#         Fallback NLU analysis when Gemini is not available
        
#         Args:
#             text: Text to analyze
            
#         Returns:
#             Simulated NLU analysis results
#         """
#         # Simple keyword extraction
#         keywords = []
#         text_lower = text.lower()
        
#         financial_keywords = [
#             'money', 'budget', 'savings', 'expenses', 'income', 'debt',
#             'investment', 'tax', 'retirement', 'emergency fund', 'credit',
#             'loan', 'mortgage', 'insurance', 'spending', 'cost', 'price'
#         ]
        
#         for keyword in financial_keywords:
#             if keyword in text_lower:
#                 keywords.append({'text': keyword, 'relevance': 0.8})
        
#         # Simple sentiment analysis
#         positive_words = ['good', 'great', 'excellent', 'improve', 'better', 'saving', 'profit']
#         negative_words = ['bad', 'worse', 'debt', 'loss', 'expensive', 'struggle', 'problem']
        
#         positive_count = sum(1 for word in positive_words if word in text_lower)
#         negative_count = sum(1 for word in negative_words if word in text_lower)
        
#         if positive_count > negative_count:
#             sentiment = {'label': 'positive', 'score': 0.7}
#         elif negative_count > positive_count:
#             sentiment = {'label': 'negative', 'score': 0.6}
#         else:
#             sentiment = {'label': 'neutral', 'score': 0.5}
        
#         # Simple entity extraction
#         entities = []
#         if any(word in text_lower for word in ['$', 'rs', '₹', 'dollar', 'rupee']):
#             entities.append({'text': 'amount', 'type': 'CURRENCY', 'score': 0.8})
        
#         # Categories
#         categories = []
#         if any(word in text_lower for word in ['budget', 'expenses', 'spending']):
#             categories.append({'label': 'Budget Management', 'score': 0.9})
#         if any(word in text_lower for word in ['savings', 'investment', 'retirement']):
#             categories.append({'label': 'Savings & Investment', 'score': 0.8})
#         if any(word in text_lower for word in ['debt', 'loan', 'credit']):
#             categories.append({'label': 'Debt Management', 'score': 0.7})
        
#         return {
#             'sentiment': sentiment,
#             'keywords': keywords[:5],
#             'entities': entities,
#             'categories': categories
#         }
    
#     def generate_response(self, prompt: str, persona: str = "general") -> str:
#         """
#         Generate response using Gemini model with intelligent fallback
        
#         Args:
#             prompt: Input prompt for the AI model
#             persona: User persona (general, student, professional)
            
#         Returns:
#             Generated response text
#         """
#         print(f"Generating response for persona: {persona}")
        
#         # Try to use Gemini first
#         if self.gemini_available:
#             try:
#                 return self._generate_with_gemini(prompt, persona)
#             except Exception as e:
#                 print(f"Error generating with Gemini: {e}")
#                 print("Falling back to enhanced rule-based response")
        
#         # Final fallback to enhanced rule-based responses
#         return self._enhanced_response_generation(prompt, persona)
    
#     def _generate_with_gemini(self, prompt: str, persona: str = "general") -> str:
#         """
#         Generate response using Gemini 2.0 Flash model
        
#         Args:
#             prompt: Input prompt
#             persona: User persona
            
#         Returns:
#             Generated response text
#         """
#         try:
#             # Create persona-specific context
#             persona_context = self._get_persona_context(persona)
            
#             # Create system instruction for financial advisor role
#             system_instruction = f"""You are a professional financial advisor with expertise in personal finance, budgeting, investments, and debt management. 
#             {persona_context}
            
#             Provide clear, actionable, and practical financial advice that is:
#             - Specific and personalized to the user's situation
#             - Actionable with concrete steps
#             - Encouraging and supportive
#             - Based on sound financial principles
#             - Easy to understand without financial jargon
#             - Focused on long-term financial health
            
#             Structure your response to be helpful and comprehensive."""
            
#             # Prepare the request payload for Gemini
#             payload = {
#                 "contents": [
#                     {
#                         "parts": [
#                             {"text": f"{system_instruction}\n\nUser Question: {prompt}"}
#                         ]
#                     }
#                 ],
#                 "generationConfig": {
#                     "temperature": 0.3,
#                     "topK": 40,
#                     "topP": 0.8,
#                     "maxOutputTokens": 800,
#                 }
#             }
            
#             # Make API request to Gemini
#             headers = {
#                 "Content-Type": "application/json"
#             }
            
#             url = f"{self.gemini_api_url}?key={self.gemini_api_key}"
#             response = requests.post(url, headers=headers, json=payload, timeout=30)
            
#             if response.status_code == 200:
#                 result = response.json()
#                 if 'candidates' in result and len(result['candidates']) > 0:
#                     generated_text = result['candidates'][0]['content']['parts'][0]['text']
                    
#                     # Validate and clean the response
#                     cleaned_response = self._clean_gemini_response(generated_text)
#                     if self._is_valid_financial_response(cleaned_response, prompt):
#                         return cleaned_response
#                     else:
#                         print("Gemini response quality check failed, using fallback")
#                         return self._enhanced_response_generation(prompt, persona)
#                 else:
#                     raise Exception("No candidates in Gemini response")
#             else:
#                 raise Exception(f"Gemini API error: {response.status_code} - {response.text}")
                
#         except Exception as e:
#             print(f"Error in Gemini generation: {e}")
#             # Fallback to enhanced response generation
#             return self._enhanced_response_generation(prompt, persona)
    
#     def _clean_gemini_response(self, response: str) -> str:
#         """
#         Clean and format the response from Gemini model
        
#         Args:
#             response: Raw response from model
            
#         Returns:
#             Cleaned response text
#         """
#         # Remove any markdown formatting if present
#         response = response.replace('**', '').replace('*', '').replace('#', '')
        
#         # Clean up whitespace and formatting
#         response = response.strip()
        
#         # Remove incomplete sentences at the end
#         sentences = response.split('. ')
#         if len(sentences) > 1 and not sentences[-1].endswith(('.', '!', '?')):
#             sentences = sentences[:-1]
#         response = '. '.join(sentences)
        
#         # Ensure proper sentence ending
#         if response and not response.endswith(('.', '!', '?')):
#             response += '.'
        
#         return response
    
#     def _is_valid_financial_response(self, response: str, original_prompt: str) -> bool:
#         """
#         Validate if the generated response is appropriate for financial advice
        
#         Args:
#             response: Generated response
#             original_prompt: Original user question
            
#         Returns:
#             True if response is valid financial advice
#         """
#         response_lower = response.lower()
#         prompt_lower = original_prompt.lower()
        
#         # Check minimum length
#         if len(response.strip()) < 30:
#             return False
        
#         # Check if response contains financial keywords or advice
#         financial_indicators = [
#             'budget', 'save', 'saving', 'money', 'income', 'expense', 'debt',
#             'investment', 'financial', 'plan', 'goal', 'recommend', 'consider',
#             'reduce', 'increase', 'strategy', 'fund', 'account', 'loan'
#         ]
        
#         has_financial_content = any(word in response_lower for word in financial_indicators)
        
#         # Check if response is coherent (not too repetitive or fragmented)
#         sentences = response.split('.')
#         if len(sentences) < 1:  # Too short or not structured
#             return False
        
#         # Check for excessive repetition
#         words = response_lower.split()
#         if len(set(words)) < len(words) * 0.6:  # Too much repetition
#             return False
        
#         # Check if response seems relevant to the question
#         question_keywords = [word for word in prompt_lower.split() if len(word) > 3]
#         response_keywords = [word for word in response_lower.split() if len(word) > 3]
        
#         # At least some keyword overlap or financial content
#         keyword_overlap = any(qw in response_keywords for qw in question_keywords[:5])
        
#         return has_financial_content and (keyword_overlap or len(response) > 100)
    
#     def _get_persona_context(self, persona: str) -> str:
#         """Get context based on user persona"""
#         contexts = {
#             "student": """You are a financial advisor specializing in helping students manage their finances. 
#             Focus on budgeting, student loans, part-time work, and building good financial habits early. 
#             Be encouraging and provide practical, actionable advice for students with limited income.""",
            
#             "professional": """You are a financial advisor for working professionals. 
#             Focus on retirement planning, investment strategies, tax optimization, and wealth building. 
#             Provide sophisticated advice while considering career growth and long-term financial goals.""",
            
#             "general": """You are a knowledgeable and friendly financial advisor. 
#             Provide clear, practical advice on personal finance topics including budgeting, saving, 
#             investing, debt management, and financial planning. Be encouraging and actionable in your responses."""
#         }
        
#         return contexts.get(persona, contexts["general"])
    
#     def _enhanced_response_generation(self, prompt: str, persona: str = "general") -> str:
#         """
#         Enhanced rule-based response generation with persona awareness
        
#         Args:
#             prompt: Input prompt
#             persona: User persona
            
#         Returns:
#             High-quality financial advice response
#         """
#         prompt_lower = prompt.lower()
        
#         # Student-specific financial advice
#         if 'student' in prompt_lower or persona == 'student':
#             if any(word in prompt_lower for word in ['loan', 'debt', 'payment']):
#                 return self._get_student_loan_advice(prompt_lower)
#             elif any(word in prompt_lower for word in ['save', 'saving', 'money']):
#                 return self._get_student_saving_advice(prompt_lower)
#             elif any(word in prompt_lower for word in ['budget', 'expense']):
#                 return self._get_student_budget_advice(prompt_lower)
#             elif any(word in prompt_lower for word in ['gym', 'fitness', 'membership']):
#                 return self._get_student_gym_advice(prompt_lower)
        
#         # Professional-specific advice
#         elif persona == 'professional':
#             if any(word in prompt_lower for word in ['investment', 'invest', 'portfolio']):
#                 return self._get_professional_investment_advice(prompt_lower)
#             elif any(word in prompt_lower for word in ['retirement', '401k', 'pension']):
#                 return self._get_professional_retirement_advice(prompt_lower)
#             elif any(word in prompt_lower for word in ['tax', 'deduction']):
#                 return self._get_professional_tax_advice(prompt_lower)
        
#         # General financial topics
#         if 'budget' in prompt_lower and 'summary' in prompt_lower:
#             return self._fallback_response_generation(prompt, persona)
#         elif any(word in prompt_lower for word in ['save', 'saving', 'money']):
#             return self._get_general_saving_advice(prompt_lower)
#         elif any(word in prompt_lower for word in ['debt', 'loan', 'credit']):
#             return self._get_debt_management_advice(prompt_lower)
#         elif any(word in prompt_lower for word in ['investment', 'invest']):
#             return self._get_investment_advice(prompt_lower)
#         elif any(word in prompt_lower for word in ['budget', 'expense']):
#             return self._get_budget_advice(prompt_lower)
#         elif any(word in prompt_lower for word in ['emergency', 'fund']):
#             return self._get_emergency_fund_advice(prompt_lower)
        
#         # Default general advice
#         return self._fallback_response_generation(prompt, persona)
    
#     def _fallback_response_generation(self, prompt: str, persona: str = "general") -> str:
#         """
#         Fallback response generation when Gemini is not available
        
#         Args:
#             prompt: Input prompt
#             persona: User persona
            
#         Returns:
#             Generated response text
#         """
#         # Simple rule-based responses for common financial topics
#         prompt_lower = prompt.lower()
        
#         if 'budget' in prompt_lower and 'summary' in prompt_lower:
#             return """**Budget Summary Analysis**

# Based on your financial data, here's my assessment:

# **Overall Financial Health**: Your budget shows a [positive/neutral/concerning] financial situation.

# **Key Insights**:
# - Your total expenses represent [X]% of your income
# - You have [positive/negative] disposable income each month
# - Your savings goal is [achievable/challenging] given current spending

# **Top Spending Categories**:
# 1. [Category 1]: [X]% of total expenses
# 2. [Category 2]: [X]% of total expenses

# **Recommendations**:
# 1. Consider reducing spending in [specific category]
# 2. Set up automatic transfers to savings
# 3. Review discretionary expenses monthly
# 4. Build an emergency fund if you haven't already

# **Risk Assessment**: [Any concerning patterns or red flags]

# Remember to regularly review and adjust your budget as your circumstances change."""

#         elif 'grocer' in prompt_lower and 'saving' in prompt_lower:
#             return """**Smart Grocery Savings Strategies**

# Here are proven ways to reduce your grocery expenses:

# **Planning & Preparation**:
# 1. **Meal Planning**: Plan your meals for the week before shopping
# 2. **Shopping List**: Make a detailed list and stick to it
# 3. **Budget Setting**: Set a weekly/monthly grocery budget
# 4. **Inventory Check**: Check what you already have at home

# **Shopping Strategies**:
# - **Store Comparison**: Compare prices across different stores
# - **Generic Brands**: Buy store brands instead of name brands (save 20-30%)
# - **Bulk Buying**: Purchase non-perishables in bulk when on sale
# - **Seasonal Shopping**: Buy fruits and vegetables in season
# - **Cash/Card**: Use cash to avoid overspending

# **Money-Saving Tips**:
# - **Coupons & Apps**: Use store apps, digital coupons, and cashback apps
# - **Sales Timing**: Shop during weekly sales and clearance events
# - **Unit Prices**: Compare cost per unit, not just package price
# - **Avoid Shopping Hungry**: Eat before grocery shopping
# - **Loyalty Programs**: Join store loyalty programs for discounts

# **Smart Food Choices**:
# - **Cook at Home**: Reduce dining out and takeaway orders
# - **Batch Cooking**: Prepare large portions and freeze extras
# - **Protein Alternatives**: Include affordable proteins like beans, eggs, chicken
# - **Frozen/Canned**: Buy frozen vegetables and canned goods when fresh is expensive

# **Expected Savings**: With these strategies, you can reduce grocery bills by 15-30%.

# Remember: Small changes in grocery habits can lead to significant savings over time!"""
        
#         elif 'house' in prompt_lower and ('saving' in prompt_lower or 'buy' in prompt_lower):
#             return """**Saving for a House Purchase**

# Buying a house worth 5 crore (₹50 million) is a significant investment. Here's a strategic approach:

# **Down Payment Planning**:
# - Traditional down payment: 10-20% = ₹5-10 million
# - Consider starting with a smaller target if this is your first home
# - Factor in additional costs: registration, stamp duty, legal fees (~2-3% extra)

# **Savings Strategy for Large Home Purchase**:
# 1. **Set a realistic timeline**: For ₹5-10 million down payment, plan 5-10 years
# 2. **Monthly savings target**: ₹50,000-₹1,00,000 per month
# 3. **High-yield investments**: Consider equity mutual funds, ELSS, PPF
# 4. **Systematic Investment Plans (SIPs)**: Automate your savings

# **Financial Preparation**:
# - **Income requirement**: Your monthly income should be 3-4x the EMI
# - **Credit score**: Maintain 750+ for better loan terms
# - **Debt-to-income ratio**: Keep below 40% including the new home loan
# - **Emergency fund**: Maintain 6 months of expenses separately

# **Alternative Approaches**:
# - Consider starting with a smaller home and upgrading later
# - Look into pre-approved loans to understand your borrowing capacity
# - Explore different locations for better value
# - Consider ready-to-move vs under-construction properties

# **Smart Tips**:
# - Track real estate market trends in your target area
# - Factor in maintenance costs (1-2% of property value annually)
# - Consider tax benefits under Section 80C and 24(b)

# Remember: A house is both a home and an investment. Plan carefully and don't overstretch your finances."""
        
#         elif 'saving' in prompt_lower or 'investment' in prompt_lower:
#             return """**Savings and Investment Guidance**

# Here are some practical steps to improve your financial situation:

# **Immediate Actions**:
# 1. **Emergency Fund**: Aim to save 3-6 months of expenses
# 2. **Automate Savings**: Set up automatic transfers from checking to savings
# 3. **Track Spending**: Use apps or spreadsheets to monitor expenses
# 4. **Reduce Expenses**: Look for areas to cut back on non-essential spending

# **Investment Considerations**:
# - Start with employer retirement plans (401k, 403b)
# - Consider Roth IRA for tax-free growth
# - Diversify investments across different asset classes
# - Start early to benefit from compound interest

# **Smart Saving Strategies**:
# - Follow the 50/30/20 rule (needs/wants/savings)
# - Use high-yield savings accounts
# - Consider CD ladders for short-term goals
# - Review and adjust your strategy regularly

# Remember: Consistency is key. Even small amounts saved regularly can grow significantly over time."""

#         elif 'debt' in prompt_lower or 'loan' in prompt_lower:
#             return """**Debt Management Strategy**

# Here's a systematic approach to managing your debt:

# **Debt Assessment**:
# 1. List all debts with balances, interest rates, and minimum payments
# 2. Calculate your total debt-to-income ratio
# 3. Identify which debts have the highest interest rates

# **Debt Repayment Strategies**:
# - **Avalanche Method**: Pay off highest interest debt first
# - **Snowball Method**: Pay off smallest balances first for motivation
# - **Debt Consolidation**: Consider combining multiple debts into one loan

# **Immediate Steps**:
# 1. Stop taking on new debt
# 2. Pay more than minimum payments when possible
# 3. Negotiate lower interest rates with creditors
# 4. Consider balance transfer cards for high-interest debt

# **Long-term Prevention**:
# - Build emergency fund to avoid future debt
# - Live below your means
# - Use credit cards responsibly
# - Save for major purchases instead of financing

# Remember: Getting out of debt takes time and discipline, but it's achievable with a solid plan."""

#         else:
#             return """**Personal Finance Advice**

# Thank you for your question! Here are some general financial principles to consider:

# **Financial Foundation**:
# 1. **Budget**: Track income and expenses to understand your cash flow
# 2. **Emergency Fund**: Save 3-6 months of expenses for unexpected events
# 3. **Debt Management**: Prioritize high-interest debt repayment
# 4. **Insurance**: Protect yourself and your family with appropriate coverage

# **Smart Money Habits**:
# - Pay yourself first (automate savings)
# - Live below your means
# - Avoid lifestyle inflation
# - Regularly review and adjust your financial plan

# **Investment Basics**:
# - Start early to benefit from compound interest
# - Diversify your investments
# - Consider your risk tolerance and time horizon
# - Take advantage of employer retirement plans

# **Continuous Learning**:
# - Stay informed about personal finance topics
# - Seek professional advice when needed
# - Learn from your financial mistakes
# - Set clear, achievable financial goals

# Remember: Financial success is a journey, not a destination. Small, consistent actions today will lead to significant results over time."""

#     def get_service_status(self) -> Dict[str, bool]:
#         """
#         Get the status of AI services
        
#         Returns:
#             Dictionary with service availability status
#         """
#         return {
#             'gemini_available': self.gemini_available,
#             'fallback_mode': not self.gemini_available
#         }
    
#     # Specific advice methods for different personas and scenarios
#     def _get_student_loan_advice(self, prompt: str) -> str:
#         """Provide student-specific advice for loans and debt"""
#         if 'gym' in prompt:
#             return """**Student Financial Advice: Balancing Gym Membership and Student Loans**

# As a student with student loans, here's my advice regarding gym expenses:

# **Priority Assessment**:
# 1. **Student loans first**: These typically have higher interest rates and longer-term impact
# 2. **Health is important**: But look for cost-effective alternatives

# **Smart Fitness Strategies for Students**:
# - **University gym**: Use your student recreation center (often included in fees)
# - **Budget gyms**: Planet Fitness, LA Fitness student discounts (~$10-15/month)
# - **Free alternatives**: YouTube workouts, running, bodyweight exercises
# - **Seasonal approach**: Outdoor activities in good weather, gym in winter

# **Financial Balance**:
# - If gym costs >$30/month, consider alternatives
# - Allocate extra income to loan payments first
# - Build an emergency fund of $500-1000
# - Track all expenses to see where your money goes

# **Long-term perspective**: Paying off loans early saves more money than most gym memberships cost. Consider free fitness options during your highest debt period.
# """
        
#         return """**Student Loan Management Strategy**

# Here's how to handle your student loans effectively:

# **Understanding Your Loans**:
# 1. List all loans with balances, interest rates, servicers
# 2. Know the difference between federal and private loans
# 3. Understand your grace period and repayment options

# **Repayment Strategies**:
# - **Standard repayment**: Highest monthly payment, least interest overall
# - **Income-driven plans**: Lower payments based on income (federal loans)
# - **Avalanche method**: Pay minimums on all, extra on highest interest rate
# - **Snowball method**: Pay smallest balance first for motivation

# **Student-Specific Tips**:
# - Keep federal loans separate from private (better protections)
# - Consider auto-pay discounts (usually 0.25% rate reduction)
# - Don't ignore loans - contact servicer if having trouble
# - Explore forgiveness programs for public service careers

# **While in School**:
# - Pay interest on unsubsidized loans if possible
# - Avoid borrowing more than necessary
# - Look for scholarships and grants continuously

# Remember: Student loans are an investment in your future earning potential.
# """
    
#     def _get_student_saving_advice(self, prompt: str) -> str:
#         """Provide student-specific saving advice"""
#         return """**Student Saving Strategies**

# Saving money as a student requires creativity and discipline:

# **Smart Saving Tactics**:
# 1. **The $1 rule**: Save every $1 bill you receive
# 2. **Round-up savings**: Round purchases up, save the difference
# 3. **Meal prep**: Cook in bulk, avoid dining out frequently
# 4. **Textbook savings**: Buy used, rent, or use library reserves
# 5. **Student discounts**: Always ask - many businesses offer them

# **Income Opportunities**:
# - Work-study jobs on campus
# - Tutoring (often pays $15-25/hour)
# - Freelance skills (writing, design, coding)
# - Paid internships and co-ops
# - Sell items you no longer need

# **Emergency Fund for Students**:
# - Start with $300-500 goal
# - Keep it in a separate savings account
# - Use only for true emergencies (car repair, medical)

# **Long-term Perspective**:
# - Build good financial habits now
# - Learn to live below your means
# - Understand wants vs needs
# - Start credit building responsibly

# Even saving $25/month as a student builds valuable habits and provides a financial cushion.
# """
    
#     def _get_student_budget_advice(self, prompt: str) -> str:
#         """Provide student-specific budgeting advice"""
#         return """**Student Budgeting Guide**

# Creating a budget as a student with irregular income:

# **Student Budget Categories**:
# - **Fixed costs**: Tuition, rent, insurance, loan payments
# - **Variable necessities**: Food, gas, school supplies
# - **Discretionary**: Entertainment, dining out, subscriptions

# **Budgeting on Irregular Income**:
# 1. Track income for 3 months to find your average
# 2. Budget based on your lowest month
# 3. Save excess from higher-income months
# 4. Use the envelope method for cash categories

# **Student-Specific Tips**:
# - **Textbooks**: Budget $400-600/semester, then find savings
# - **Food**: Meal plans vs. cooking - calculate the real cost
# - **Transportation**: Consider walking/biking vs. car costs
# - **Free entertainment**: Campus events, free museum days

# **Track These Categories**:
# - Housing (aim for <30% of income)
# - Food (15-20%)
# - Transportation (10-15%)
# - School supplies (5-10%)
# - Savings (at least 5% even if small amounts)

# **Tools**: Use apps like Mint, YNAB (free for students), or simple spreadsheets.

# Remember: Your student years are for learning financial discipline that will serve you throughout life.
# """
    
#     def _get_student_gym_advice(self, prompt: str) -> str:
#         """Specific advice about gym expenses for students"""
#         return self._get_student_loan_advice(prompt)  # Reuse the gym-specific advice
    
#     def _get_professional_investment_advice(self, prompt: str) -> str:
#         """Investment advice for working professionals"""
#         return """**Professional Investment Strategy**

# As a working professional, here's how to approach investing:

# **Investment Priority Order**:
# 1. **Emergency fund**: 3-6 months expenses in high-yield savings
# 2. **Employer 401(k) match**: Free money - contribute enough to get full match
# 3. **High-interest debt**: Pay off credit cards (>6% interest)
# 4. **Max retirement accounts**: 401(k), IRA ($6,500 limit for 2024)
# 5. **Taxable investment accounts**: For additional long-term growth

# **Asset Allocation by Age**:
# - **20s-30s**: 80-90% stocks, 10-20% bonds
# - **40s**: 70-80% stocks, 20-30% bonds
# - **50s+**: 60-70% stocks, 30-40% bonds
# - Rule of thumb: (120 - your age) = % in stocks

# **Professional Investment Vehicles**:
# - **Index funds**: Low fees, broad diversification (VTI, VXUS)
# - **Target-date funds**: Automatic rebalancing
# - **ETFs**: Tax-efficient, liquid
# - **Real estate**: REITs or rental properties for diversification

# **Advanced Strategies**:
# - Tax-loss harvesting in taxable accounts
# - Backdoor Roth IRA if high income
# - Mega backdoor Roth if available
# - Consider professional financial advisor for complex situations

# Start early, invest consistently, and keep fees low for long-term wealth building.
# """
    
#     def _get_professional_retirement_advice(self, prompt: str) -> str:
#         """Retirement planning advice for professionals"""
#         return """**Professional Retirement Planning**

# Maximize your retirement savings as a working professional:

# **Retirement Account Limits (2024)**:
# - **401(k)**: $23,000 ($30,500 if 50+)
# - **IRA**: $7,000 ($8,000 if 50+)
# - **Total possible**: $30,000+ annually

# **Employer Benefits to Maximize**:
# - Get full 401(k) match (typically 3-6% of salary)
# - HSA if available ($4,300 individual, $8,550 family)
# - Consider after-tax 401(k) contributions if mega backdoor available

# **Professional Retirement Strategies**:
# 1. **Automate everything**: Set up automatic contributions
# 2. **Increase with raises**: Boost contribution % with each promotion
# 3. **Tax diversification**: Mix of traditional and Roth accounts
# 4. **Rebalance annually**: Maintain target asset allocation

# **Income-Specific Considerations**:
# - **High earners**: May be limited from Roth IRA (backdoor option)
# - **Variable income**: Contribute heavily in high-earning years
# - **Stock options**: Understand vesting and tax implications

# **Retirement Planning Milestones**:
# - Age 30: 1x annual salary saved
# - Age 40: 3x annual salary
# - Age 50: 6x annual salary
# - Age 60: 8x annual salary
# - Retirement: 10-12x annual salary

# Consider working with a fee-only financial advisor for comprehensive planning.
# """
    
#     def _get_professional_tax_advice(self, prompt: str) -> str:
#         """Tax optimization advice for professionals"""
#         return """**Professional Tax Optimization**

# Strategies to minimize your tax burden legally:

# **Pre-Tax Savings (Reduces Current Taxes)**:
# - 401(k) contributions
# - Traditional IRA (if eligible)
# - HSA contributions (triple tax advantage)
# - Flexible Spending Account (FSA)

# **Tax Credits vs. Deductions**:
# - **Credits**: Dollar-for-dollar tax reduction (Child Tax Credit, Education Credits)
# - **Deductions**: Reduce taxable income (mortgage interest, charitable donations)

# **Professional Tax Strategies**:
# 1. **Maximize retirement contributions**: Immediate tax savings
# 2. **Tax-loss harvesting**: Offset gains with losses in taxable accounts
# 3. **Charitable giving**: Bunching donations, donor-advised funds
# 4. **Professional development**: Many education expenses are deductible

# **Advanced Strategies**:
# - **Backdoor Roth**: If income too high for regular Roth
# - **Mega backdoor Roth**: If employer plan allows
# - **Tax-efficient fund placement**: Bonds in tax-advantaged accounts
# - **Municipal bonds**: For high earners in high-tax states

# **Business Owners Additional Options**:
# - SEP-IRA or Solo 401(k)
# - Business expense deductions
# - Qualified Business Income (QBI) deduction

# **Important**: Tax laws change frequently. Consider consulting a tax professional for complex situations or significant income changes.

# Proactive tax planning can save thousands annually.
# """
    
#     def _get_general_saving_advice(self, prompt: str) -> str:
#         """General saving advice for all personas"""
#         return self._fallback_response_generation(prompt, "general")
    
#     def _get_debt_management_advice(self, prompt: str) -> str:
#         """General debt management advice"""
#         return self._fallback_response_generation(prompt, "general")
    
#     def _get_investment_advice(self, prompt: str) -> str:
#         """General investment advice"""
#         return self._fallback_response_generation(prompt, "general")
    
#     def _get_budget_advice(self, prompt: str) -> str:
#         """General budget advice"""
#         return self._fallback_response_generation(prompt, "general")
    
#     def _get_emergency_fund_advice(self, prompt: str) -> str:
#         """Emergency fund advice"""
#         return """**Emergency Fund Essentials**

# Building your financial safety net:

# **Emergency Fund Basics**:
# - **Purpose**: Cover unexpected expenses without debt
# - **Amount**: 3-6 months of essential expenses
# - **Location**: High-yield savings account (accessible but separate)

# **How Much Do You Need?**:
# - **Single, stable job**: 3-4 months expenses
# - **Married, dual income**: 3-4 months expenses
# - **Single earner household**: 6 months expenses
# - **Variable income**: 6+ months expenses
# - **High-risk job**: 6+ months expenses

# **Building Your Emergency Fund**:
# 1. **Start small**: Even $500 helps with minor emergencies
# 2. **Automate**: Set up automatic weekly/monthly transfers
# 3. **Use windfalls**: Tax refunds, bonuses, gifts
# 4. **Side income**: Temporarily direct extra earnings to fund

# **What Qualifies as an Emergency?**:
# ✅ **True emergencies**: Job loss, medical bills, major car/home repairs
# ❌ **Not emergencies**: Vacations, shopping, known upcoming expenses

# **Where to Keep It**:
# - High-yield savings account (current rates 4-5%)
# - Money market account
# - Short-term CDs (if you have multiple months saved)
# - NOT: Checking account, investment accounts, cash at home

# **After It's Built**:
# - Only use for true emergencies
# - Replenish immediately after use
# - Review annually and adjust for lifestyle changes

# Your emergency fund provides peace of mind and financial stability.
# """