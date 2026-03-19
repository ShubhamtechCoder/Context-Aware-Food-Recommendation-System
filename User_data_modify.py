import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

# FOOD CATALOG

food_catalog = {
    "Fast_Food": ["Pizza","Burger","Fries","Garlic Bread","Sandwich","Nachos","Cheesy Dip"],
    "North_Indian": ["Biryani","Roti","Paneer Tikka","Dal Makhani","Raita","Salad","Papad"],
    "South_Indian": ["Dosa","Idli","Vada","Sambar","Upma","Chutney"],
    "Chinese": ["Hakka Noodles","Fried Rice","Manchurian","Spring Roll","Chilli Paneer"],
    "Desserts": ["Gulab Jamun","Rasgulla","Brownie","Ice Cream","Kheer","Choco Lava Cake"],
    "Beverages": ["Coke","Pepsi","Lassi","Milkshake","Tea","Cold Coffee"]
}

cuisines = list(food_catalog.keys())

# USERS

num_users = 200
users = []

for u in range(num_users):
    pref = random.choice(cuisines)
    users.append({
        "user_id": u,
        "preferred_category": pref,
        "user_region": random.choice(["South_Delhi","West_Delhi","Gurgaon","Noida","Navi_Mumbai"]),
        "avg_cart_item_count": np.random.choice([1,2,3,4], p=[0.4,0.35,0.2,0.05])
    })

users_df = pd.DataFrame(users)

# GENERATE ROWS

rows = []
N = 1200

for i in range(N):
    
    user = users_df.sample(1).iloc[0]
    
    preferred_category = user.preferred_category
    top_ordered_item = random.choice(food_catalog[preferred_category])
    
    order_party_type = "group" if user.avg_cart_item_count>=3 else "solo"
    
    restaurant_cuisine_type = random.choice(cuisines)
    restaurant_item_popularity = round(np.random.beta(2,5),3)
    
    cart_item_count = np.random.choice([1,2,3,4], p=[0.35,0.35,0.2,0.1])
    cart_total_value = int(np.random.normal(250 + cart_item_count*80,40))
    cart_cuisine_type = preferred_category
    
    meal_time_segment = np.random.choice(["breakfast","lunch","snacks","dinner"], p=[0.1,0.4,0.2,0.3])
    day_of_week_type = np.random.choice(["weekday","weekend"], p=[0.7,0.3])
    occasion_type = np.random.choice(["none","festival","birthday","anniversary"], p=[0.8,0.1,0.05,0.05])
    seasonal_preference = random.choice(["summer","winter","monsoon"])
    
    user_item_affinity = round(np.random.beta(2,4),3)
    item_cooccurrence_score = int(np.random.exponential(3))
    user_category_affinity = round(np.random.beta(3,3),3)
    
    item_price_tier = np.random.choice(["low","medium","high"], p=[0.4,0.4,0.2])
    dietary_type = np.random.choice(["veg","nonveg","dairy","egg"], p=[0.5,0.3,0.15,0.05])
    
    base_prob = 0.15
    if item_cooccurrence_score>3: base_prob += 0.35
    if user_category_affinity>0.6: base_prob += 0.2
    if meal_time_segment=="dinner": base_prob += 0.1
    
    label_addon_added = int(np.random.rand()<min(base_prob,0.9))
    
    rows.append({
        "user_id": user.user_id,
        "top_ordered_item": top_ordered_item,
        "avg_cart_item_count": user.avg_cart_item_count,
        "preferred_category": preferred_category,
        "order_party_type": order_party_type,
        "user_region": user.user_region,
        "restaurant_cuisine_type": restaurant_cuisine_type,
        "restaurant_item_popularity": restaurant_item_popularity,
        "cart_total_value": cart_total_value,
        "cart_item_count": cart_item_count,
        "cart_cuisine_type": cart_cuisine_type,
        "meal_time_segment": meal_time_segment,
        "day_of_week_type": day_of_week_type,
        "occasion_type": occasion_type,
        "seasonal_preference": seasonal_preference, 
        "user_item_affinity": user_item_affinity,
        "item_cooccurrence_score": item_cooccurrence_score,
        "user_category_affinity": user_category_affinity,
        "item_price_tier": item_price_tier,
        "dietary_type": dietary_type,
        "label_addon_added": label_addon_added
    })

df_users = pd.DataFrame(rows)
df_users.to_csv("csao_personalized_dataset.csv", index=False)

print(df_users.head())