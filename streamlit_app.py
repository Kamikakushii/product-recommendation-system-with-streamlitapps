import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load dataset
df = pd.read_csv("products.csv")
print("Dataset loaded successfully!")
print(df)  # Debug print

# Function to recommend products
def recommend_products(product_name, df, top_n=5):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['product_description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(df.index, index=df['product_name']).drop_duplicates()

    try:
        idx = indices[product_name]
    except KeyError:
        return []  # Product not found

    # Get category of the input product
    product_category = df.loc[idx, "category"]

    # Compute similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get product indices and filter by category
    product_indices = [i[0] for i in sim_scores if df.iloc[i[0]]["category"] == product_category]
    return df['product_name'].iloc[product_indices][:top_n]

# Streamlit app
st.title("Product Recommendation System")
st.write("Enter a product name to get recommendations.")

# Input field for product name
product_name = st.text_input("Product Name:")

# Button to get recommendations
if st.button("Get Recommendations"):
    if product_name:
        print(f"User entered product: {product_name}")  # Debug print
        recommendations = recommend_products(product_name, df)
        if len(recommendations) == 0:
            st.write("No recommendations found. Please try another product.")
        else:
            st.write("Recommended Products:")
            for product in recommendations:
                st.write(f"- {product}")
    else:
        st.write("Please enter a product name.")