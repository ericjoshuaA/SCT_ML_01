import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd

def generate_synthetic_house_data(num_samples=1000):
    """
    Generates a synthetic dataset for house prices.
    Args:
        num_samples (int): The number of data samples to generate.
    Returns:
        pd.DataFrame: A DataFrame containing 'SquareFootage', 'Bedrooms', 'Bathrooms', and 'Price' columns.
    """
    np.random.seed(42)  # for reproducibility
    # Generate features
    square_footage = np.random.normal(loc=1800, scale=500, size=num_samples)
    square_footage = np.maximum(500, square_footage)  # Ensure minimum square footage
    bedrooms = np.random.randint(1, 6, size=num_samples)  # 1 to 5 bedrooms
    bathrooms = np.random.uniform(1, 4, size=num_samples)  # 1 to 3.5 bathrooms
    bathrooms = np.round(bathrooms * 2) / 2  # Round to nearest 0.5

    # Define a base price and add noise
    # Price = (SqFt * weight_sqft) + (Bedrooms * weight_bedrooms) + (Bathrooms * weight_bathrooms) + bias + noise
    price_base = (
        square_footage * 150 +
        bedrooms * 25000 +
        bathrooms * 15000 +
        100000  # Base price offset
    )
    # Add some random noise to make it more realistic
    price = price_base + np.random.normal(loc=0, scale=50000, size=num_samples)
    price = np.maximum(50000, price)  # Ensure minimum price

    data = pd.DataFrame({
        'SquareFootage': square_footage,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Price': price
    })
    return data

def train_and_evaluate_model(data):
    """
    Trains a linear regression model and evaluates its performance.
    Args:
        data (pd.DataFrame): The dataset containing features and target.
    Returns:
        tuple: A tuple containing:
            - sklearn.linear_model.LinearRegression: The trained model.
            - dict: A dictionary of evaluation metrics (MAE, R2).
    """
    # Define features (X) and target (y)
    X = data[['SquareFootage', 'Bedrooms', 'Bathrooms']]
    y = data['Price']

    # Split data into training and testing sets (80% for training, 20% for testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}\n")

    # Initialize the Linear Regression model
    model = LinearRegression()

    # Train the model
    print("Training the Linear Regression model...")
    model.fit(X_train, y_train)
    print("Model training complete.\n")

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {
        'Mean Absolute Error (MAE)': mae,
        'R-squared (R2)': r2
    }
    print("Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"- {metric}: {value:.2f}")

    print(f"\nModel Coefficients:")
    print(f"- Square Footage: {model.coef_[0]:.2f}")
    print(f"- Bedrooms: {model.coef_[1]:.2f}")
    print(f"- Bathrooms: {model.coef_[2]:.2f}")
    print(f"- Intercept: {model.intercept_:.2f}\n")
    return model, metrics

def predict_house_price(model, square_footage, bedrooms, bathrooms):
    """
    Predicts the price of a house using the trained model.
    Args:
        model (sklearn.linear_model.LinearRegression): The trained model.
        square_footage (float): The square footage of the house.
        bedrooms (int): The number of bedrooms.
        bathrooms (float): The number of bathrooms.
    Returns:
        float: The predicted price of the house.
    """
    # Create a DataFrame for the new input, ensuring column names match training data
    new_house_data = pd.DataFrame([[square_footage, bedrooms, bathrooms]],
                                  columns=['SquareFootage', 'Bedrooms', 'Bathrooms'])
    predicted_price = model.predict(new_house_data)[0]
    return predicted_price

if __name__ == "__main__":
    print("--- Starting House Price Prediction Model ---")
    # 1. Generate Synthetic Data
    print("Generating synthetic house price data...")
    house_data = generate_synthetic_house_data(num_samples=2000)
    print(f"Generated {len(house_data)} samples.")
    print("Sample of generated data:")
    print(house_data.head())
    print("-" * 50)

    # 2. Train and Evaluate the Model
    trained_model, evaluation_metrics = train_and_evaluate_model(house_data)
    print("-" * 50)

    # 3. Demonstrate Predictions
    print("Demonstrating predictions for new houses:")
    # Example 1: A typical house
    sq_ft_1, bed_1, bath_1 = 2000, 3, 2.5
    price_1 = predict_house_price(trained_model, sq_ft_1, bed_1, bath_1)
    print(f"Predicted price for a {sq_ft_1} sq ft house with {bed_1} beds and {bath_1} baths: ${price_1:,.2f}")

    # Example 2: A smaller house
    sq_ft_2, bed_2, bath_2 = 1200, 2, 1.0
    price_2 = predict_house_price(trained_model, sq_ft_2, bed_2, bath_2)
    print(f"Predicted price for a {sq_ft_2} sq ft house with {bed_2} beds and {bath_2} baths: ${price_2:,.2f}")

    # Example 3: A larger house
    sq_ft_3, bed_3, bath_3 = 3500, 5, 4.0
    price_3 = predict_house_price(trained_model, sq_ft_3, bed_3, bath_3)
    print(f"Predicted price for a {sq_ft_3} sq ft house with {bed_3} beds and {bath_3} baths: ${price_3:,.2f}")

    print("\n--- House Price Prediction Model Finished ---")