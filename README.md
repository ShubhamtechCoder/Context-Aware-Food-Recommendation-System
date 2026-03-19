📌 Project Title
Context-Aware-Food-Recommendation-System

This project implements a machine learning-based recommendation system that suggests relevant add-on food items based on user behavior, cart composition, and contextual signals. The system generates real-time Top-N recommendations while ensuring constraints such as restaurant availability and dietary compatibility.

🧠 Approach
1. Data Preparation
Created a personalized behavioral dataset (1200+ rows)
Integrated a large restaurant menu dataset (90K+ records)
Simulated real-world features like:
   user-item affinity
   item co-occurrence
   category preference

2. Feature Engineering
Key features used:
   User features (preferences, order behavior)
   Cart features (items, total value)
   Context features (time, day type)
   Item features (price, dietary type, category)

3. Model
Used Random Forest Classifier for prediction
Output: probability of add-on selection
Ranked items based on predicted scores

4. Recommendation Pipeline
Candidate Generation
   Restricted to restaurant menu
   Applied dietary filters (veg/non-veg logic)

Ranking
   ML-based scoring
   Meal pairing heuristics (e.g., curry → roti, pizza → drinks)

Top-N Selection
   Returns Top 8 recommendations

5. Advanced Features
Multi-item cart handling
Context-aware recommendations
Cold-start handling:
   New users → popularity fallback
   New restaurants → category-based suggestions
Safe encoding for unseen categorical values

⚡ Performance
Model Metrics
   Accuracy: ~0.68
   AUC: ~0.70
   Precision: ~0.56


System Performance
   Inference latency: ~40–70 ms
   Generates real-time recommendations
   
🧰 Tech Stack
Python
Pandas, NumPy
Scikit-learn (Random Forest)
Data Simulation & Feature Engineering

📊 Output Example
User: 36  
Restaurant: XYZ  
Cart: Butter Chicken  

Top Recommendations:
1. Tandoori Roti  
2. Butter Naan  
3. Dal Makhani  
4. Jeera Rice  
5. Gulab Jamun  
...

Built a real-time, context-aware recommendation system combining machine learning and rule-based ranking to generate personalized food add-ons under strict business constraints.
