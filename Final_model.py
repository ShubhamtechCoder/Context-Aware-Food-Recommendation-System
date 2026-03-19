import pandas as pd
import numpy as np
import time

# 1. LOAD DATA

menu_df = pd.read_csv("csao_menu_full.csv")
behavior_df = pd.read_csv("csao_personalized_dataset.csv")

print("Menu:", menu_df.shape)
print("Behavior:", behavior_df.shape)

# 2. MERGE MENU FEATURES INTO BEHAVIOR

behavior_df = behavior_df.merge(
    menu_df,
    left_on="top_ordered_item",
    right_on="item_name",
    how="left"
)

print("Merged dataset:", behavior_df.shape)

# 3. PREPARE FEATURES

target = "label_addon_added"

drop_cols = ["item_name"]  # duplicate after merge

X = behavior_df.drop([target] + drop_cols, axis=1)
y = behavior_df[target]

# 4. ENCODE CATEGORICAL FEATURES

from sklearn.preprocessing import LabelEncoder

encoders = {}
X_encoded = X.copy()

for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

# 5. TRAIN MODEL

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=14,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 6. EVALUATION

from sklearn.metrics import accuracy_score, roc_auc_score

# 6A. RANKING METRICS

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]



print("\n=== MODEL METRICS ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred),3))
print("AUC:", round(roc_auc_score(y_test, y_prob),3))
def precision_at_k(y_true, y_prob, k=50):
    df_eval = pd.DataFrame({"label": y_true, "prob": y_prob})
    df_eval = df_eval.sort_values("prob", ascending=False)
    return df_eval.head(k)["label"].mean()

def recall_at_k(y_true, y_prob, k=50):
    df_eval = pd.DataFrame({"label": y_true, "prob": y_prob})
    df_eval = df_eval.sort_values("prob", ascending=False)
    return df_eval.head(k)["label"].sum() / df_eval["label"].sum()

import numpy as np
def ndcg_at_k(y_true, y_prob, k=50):
    df_eval = pd.DataFrame({"label": y_true, "prob": y_prob})
    df_eval = df_eval.sort_values("prob", ascending=False)
    topk = df_eval.head(k)

    dcg = np.sum((2**topk["label"] - 1) / np.log2(np.arange(2, len(topk)+2)))
    ideal = sorted(y_true, reverse=True)[:k]
    idcg = np.sum((2**np.array(ideal) - 1) / np.log2(np.arange(2, len(ideal)+2)))

    return dcg / idcg if idcg > 0 else 0

precision_k = precision_at_k(y_test, y_prob, 50)
recall_k = recall_at_k(y_test, y_prob, 50)
ndcg_k = ndcg_at_k(y_test.values, y_prob, 50)


# 6B. BUSINESS METRICS (OFFLINE ESTIMATE)

acceptance_rate = precision_k

avg_addon_value = behavior_df["cart_total_value"].mean() * 0.2
aov_lift = acceptance_rate * avg_addon_value

ctr = acceptance_rate
baseline_c2o = 0.65
c2o_lift = acceptance_rate * 0.05

# 7. FINAL CSAO RECOMMENDER

def csao_recommend_multi(user_id, restaurant_name, cart_items, top_n=8):

    start = time.time()

    # ---------- USER ----------
    user_rows = behavior_df[behavior_df["user_id"] == user_id]
    if user_rows.empty:
        print("❌ User not found")
        return

    # ---------- RESTAURANT ----------
    rest_menu = menu_df[menu_df["restaurant_name"] == restaurant_name]
    if rest_menu.empty:
        print("❌ Restaurant not found")
        return

    cart_items = [c.strip() for c in cart_items]

    # validate cart
    valid_cart = []
    for c in cart_items:
        if c in rest_menu["item_name"].values:
            valid_cart.append(c)

    if len(valid_cart) == 0:
        print("❌ No cart items found in this restaurant")
        return

    # diet rule → if ANY cart item is veg → veg only
    cart_diets = rest_menu[rest_menu["item_name"].isin(valid_cart)]["dietary_type"]

    if all(d == "Veg" for d in cart_diets):
        candidates = rest_menu[rest_menu["dietary_type"] == "Veg"].copy()
    else:
        candidates = rest_menu.copy()

    # remove cart items
    candidates = candidates[~candidates["item_name"].isin(valid_cart)]

    # ---------- MEAL PAIRING BOOST ----------
    meal_pairs = {
        "sabzi": ["naan","roti","chapati","paratha"],
        "biryani": ["raita","salad"],
        "pizza": ["garlic","bread","cola"],
        "burger": ["fries","cola"],
        "noodles": ["manchurian","spring","roll"]
    }

    def detect_group(name):
        n = name.lower()
        if any(x in n for x in ["aloo","paneer","masala","sabzi","curry"]):
            return "sabzi"
        if "biryani" in n:
            return "biryani"
        if "pizza" in n:
            return "pizza"
        if "burger" in n:
            return "burger"
        if "noodle" in n:
            return "noodles"
        return None

    boosted_items = []

    for ci in valid_cart:
        g = detect_group(ci)
        if g in meal_pairs:
            keywords = meal_pairs[g]
            boosted = candidates[
                candidates["item_name"]
                .str.lower()
                .str.contains("|".join(keywords), na=False)
            ]
            boosted_items.append(boosted)

    if boosted_items:
        boosted_df = pd.concat(boosted_items)
        candidates = pd.concat([boosted_df, candidates], ignore_index=True)

    candidates = candidates.drop_duplicates(subset=["item_name"])

    # ---------- USER CONTEXT ----------
    user_context = user_rows.iloc[0].copy()

    rows = []

    # score candidate against ALL cart items
    for _, item in candidates.iterrows():

        scores = []

        for cart in valid_cart:

            r = user_context.copy()

            r["top_ordered_item"] = item["item_name"]
            r["restaurant_name"] = item["restaurant_name"]
            r["city"] = item["city"]
            r["place"] = item["place"]
            r["cuisine_type"] = item["cuisine_type"]
            r["price"] = item["price"]
            r["delivery_rating"] = item["delivery_rating"]
            r["avg_rating"] = item["avg_rating"]
            r["is_bestseller"] = item["is_bestseller"]
            r["is_highly_rated"] = item["is_highly_rated"]
            r["is_expensive"] = item["is_expensive"]
            r["dietary_type"] = item["dietary_type"]
            r["menu_category"] = item["menu_category"]
            r["price_range"] = item["price_range"]

            rows.append(r)

    candidate_df = pd.DataFrame(rows)

    # ---------- ENCODE ----------
    encoded = candidate_df.copy()

    for col, le in encoders.items():
        if col in encoded.columns:
            known = set(le.classes_)
            encoded[col] = encoded[col].apply(
                lambda x: x if str(x) in known else le.classes_[0]
            )
            encoded[col] = le.transform(encoded[col].astype(str))

    encoded = encoded[X_encoded.columns]

    # ---------- PREDICT ----------
    probs = model.predict_proba(encoded)[:,1]
    candidate_df["probability"] = probs

    # aggregate per item (multi-cart)
    final_scores = (
        candidate_df
        .groupby("top_ordered_item")["probability"]
        .mean()
        .reset_index()
    )

    final_scores = final_scores.sort_values(
        "probability", ascending=False
    ).head(top_n)

    latency = (time.time() - start) * 1000

    print("\n=== CSAO MULTI-CART TOP-8 ===")
    print("User:", user_id)
    print("Restaurant:", restaurant_name)
    print("Cart:", valid_cart)
    print("Latency:", round(latency,2), "ms\n")

    print(final_scores)

    return final_scores

# 7. OPERATIONAL METRICS

import random, time

def measure_latency(runs=30):
    users = behavior_df["user_id"].unique()
    rests = menu_df["restaurant_name"].unique()
    items = behavior_df["top_ordered_item"].unique()

    latencies = []

    for _ in range(runs):
        u = int(random.choice(users))
        r = random.choice(rests)
        i = random.choice(items)

        start = time.time()
        csao_recommend(u, r, i)
        latencies.append((time.time()-start)*1000)

    return np.mean(latencies)

def measure_coverage(runs=50):
    users = behavior_df["user_id"].unique()
    rests = menu_df["restaurant_name"].unique()
    items = behavior_df["top_ordered_item"].unique()

    success = 0

    for _ in range(runs):
        u = int(random.choice(users))
        r = random.choice(rests)
        i = random.choice(items)

        try:
            rec = csao_recommend(u, r, i)
            if rec is not None:
                success += 1
        except:
            pass

    return success / runs

# avg_latency = measure_latency(30)
# coverage = measure_coverage(50)


# FINAL CSAO EVALUATION SUMMARY

print("\n===== CSAO FINAL METRICS =====")
print("AUC:", round(roc_auc_score(y_test, y_prob),3))
print("Precision@50:", round(precision_k,3))
print("Recall@50:", round(recall_k,3))
print("NDCG@50:", round(ndcg_k,3))

print("\nBusiness Metrics")
print("Acceptance Rate:", round(acceptance_rate,3))
print("Estimated AOV Lift:", round(aov_lift,2))
print("CTR (proxy):", round(ctr,3))
print("C2O Lift (proxy):", round(c2o_lift,3))

# ❌ DO NOT AUTO-RUN LATENCY HERE
# avg_latency = measure_latency(30)
# coverage = measure_coverage(50)

# INTERACTIVE

while True:
    print("\n--- CSAO Multi-Cart Recommender ---")

    u = input("User ID (or exit): ")
    if u.lower() == "exit":
        break

    uid = int(u)

    r = input("Restaurant: ")

    cart_input = input("Enter cart items (comma separated): ")
    cart_items = cart_input.split(",")

    csao_recommend_multi(uid, r, cart_items, 8)