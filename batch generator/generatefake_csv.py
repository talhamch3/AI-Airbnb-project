import pandas as pd
import numpy as np

np.random.seed(42)

n = 300

# Generate synthetic dataset
df = pd.DataFrame({
    "listing_id": np.arange(1, n+1),

    # Bedrooms: 1–5
    "bedrooms": np.random.randint(1, 6, n),

    # Bathrooms: 1–3
    "bathrooms": np.round(np.random.uniform(1, 3, n), 1),

    # Price depends slightly on bedrooms (realistic)
    "price": np.random.randint(50, 300, n) + np.random.randint(0, 50, n),

    # Availability rate (0 to 1)
    "availability_rate": np.round(np.random.uniform(0, 1, n), 2),

    # Review count (popularity proxy)
    "review_count": np.random.randint(0, 200, n),

    # Season (balanced)
    "season": np.random.choice(
        ["Winter", "Spring", "Summer", "Autumn"], 
        size=n
    )
})

# Save CSV
df.to_csv("data/monitoring_batch.csv", index=False)

print("✅ CSV generated: data/monitoring_batch.csv")