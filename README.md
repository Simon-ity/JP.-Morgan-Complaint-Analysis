# JP.-Morgan-Complaint-Analysis
![JPMorganChase-brown-and-white](https://github.com/user-attachments/assets/2fcbeded-b0dd-4705-acfd-a1b8869f8e27)

# 🎯 Project Overview
This project applies sentiment analysis and logistic regression to consumer financial complaints data, predicting dispute likelihood based on emotional indicators in complaint narratives.

# Data Cleaning With Excel Power Query

- Column Standardization: Used Power Query Editor to rename and format columns

- Missing Value Treatment: Applied strategic fill methods for different data types

- Categorical Mapping: Standardized response categories and consent indicators

- Export Preparation: Generated clean dataset for Python analysis

# Sentiment Analysis

- TextBlob Library: Extracted polarity and subjectivity scores
  
- Emotion Detection: Custom algorithms for anger and trust indicators
  
- Score Normalization: Scaled emotion scores to 0-1 range

# Statistical Modeling

- Logistic Regression: Binary classification for dispute prediction
  
- Feature Selection: Sentiment polarity, anger score, trust score
  
- Model Validation: Train-test split with performance evaluation

# Load and analyze data
     import pandas as pd
    from textblob import TextBlob
    from sklearn.linear_model import LogisticRegression

# Run analysis
df = pd.read_excel('cleaned_complaints.xlsx')


# Statistical Results
                         Logistic Regression Coefficients
| Feature | Coefficients | P-value  | Interpretation
| ------- | ------- |------- |------- |
| Anger Score | +0.255 | < 0.05 | Higher Anger -> Higher dispute probability |
| Trust Score | +0.043 | < 0.05 | Higher Trust -> Lower dispute probability |
| Sentiment Polarity | +0. 104 | < 0.1 | Positive sentiment -> Higher engagement |

Code

       #SIMPLE LOGISTIC REGRESSION
     # ==============================================================================
    features = ['sentiment_polarity', 'anger_score', 'trust_score']
     if 'sentiment_subjectivity' in df_text.columns:
    features.append('sentiment_subjectivity')

    df_model = df_text[features + ['dispute_binary']].dropna()

    if len(df_model) >= 10:  
    X = df_model[features]
    y = df_model['dispute_binary']
    
    # Split data
    if len(df_model) > 20:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y  
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    accuracy = model.score(X_test, y_test)
    
   
    print("KEY PREDICTORS (Feature Importance):")
    feature_importance = list(zip(features, model.coef_[0]))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for feature, coef in feature_importance:
        direction = "INCREASES" if coef > 0 else "DECREASES"
        significance = "***" if abs(coef) > 0.5 else ("**" if abs(coef) > 0.2 else "*")
        print(f"   • {feature}: {coef:+.3f} {significance}")
        print(f"     → {direction} dispute probability")

# Key Results

- Dataset: 1,048,575 total complaints analyzed

- Text Analysis: 240,881 complaints with narratives

- Model Accuracy: 99.1% dispute prediction accuracy

- Sentiment is the strongest predictor (coefficient = -0.571)
   
- More positive sentiment → Lower dispute probability
   
- Overall dispute rate: 0.7%

- Statistical Significance: Trust levels inversely correlate with dispute probability

# Top Complaint Categories

- Credit reporting services (35.5%)

- Credit card issues (11.4%)

- Debt collection (10.3%)
  
- Banking services (6.7%)

# Sentiment Analysis 
### Sentiment Distribution Analysis

<img width="450" height="450" alt="Screenshot 2025-08-02 at 12 22 35 AM" src="https://github.com/user-attachments/assets/0b4b2984-9c05-4fb7-be28-ae4b2cb4eb28" />

**Sentiment Distribution Histogram**
- **Pattern**: Sharp peak at 0.0 (neutral sentiment) with normal distribution
- **Finding**: Most JP Morgan complaints are factual and unemotional rather than angry rants
- **Range**: -0.8 to +0.6 sentiment polarity
- **Insight**: Professional, measured complaint language dominates customer feedback

### Sentiment Categories

<img width="838" height="504" alt="Screenshot 2025-08-02 at 12 39 21 AM" src="https://github.com/user-attachments/assets/845dd843-4cb2-4d12-ba38-c8992d87f9d7" />


**Sentiment Categories Pie Chart**
- **Neutral (59.2%)**: Majority of JP Morgan complaints are factual, business-like reporting
- **Positive (25.8%)**: Quarter of complaints show constructive or satisfied tone
- **Negative (15.0%)**: Smallest segment represents frustrated or dissatisfied customers
- **Insight**: 85% of complaints are neutral/positive, indicating professional customer communication

### Emotion Analysis & Dispute Patterns

<img width="320" height="320" alt="Screenshot 2025-08-02 at 12 43 04 AM" src="https://github.com/user-attachments/assets/2d5a7c2d-6cf9-4919-acfa-3c768134f0da" />
<img width="320" height="320" alt="Screenshot 2025-08-02 at 12 43 18 AM" src="https://github.com/user-attachments/assets/6ed09672-74a9-4dae-b4e5-db825908b498" />


**Left - Average Emotion Scores:**
- **Trust (0.443)**: Moderate trust levels in JP Morgan customer feedback
- **Anger (0.032)**: Very low anger scores indicate calm, professional complaints
- **Ratio**: Trust is 14x higher than anger, showing constructive customer relationships

**Right - Dispute Rate by Sentiment:**
- **Neutral sentiment**: Highest dispute rate (~0.008) - factual complaints more likely escalated
- **Negative sentiment**: Moderate dispute rate (~0.006) - emotional complaints sometimes resolved
- **Positive sentiment**: Lowest dispute rate (~0.005) - satisfied customers rarely dispute

### Anger vs Trust Relationship

<img width="375" height="416" alt="Screenshot 2025-08-02 at 1 57 58 AM" src="https://github.com/user-attachments/assets/0d81c4c8-a7d2-4461-bd43-f14b6c422313" />


**Anger vs Trust Scatter Plot**
- **Pattern**: No clear correlation between anger and trust scores
- **Distribution**: Points scattered across all combinations (0.0 to 1.0 range)
- **Finding**: Anger and trust are independent emotions in JP Morgan complaints
- **Insight**: Customers can express anger while maintaining trust, or vice versa

# Recommendations

- Early Warning System: Flag complaints with anger scores > 0.5

- Response Prioritization: Fast-track high-risk complaints

- Trust Building: Implement transparent communication protocols

- Training Programs: Develop emotional intelligence for customer service
