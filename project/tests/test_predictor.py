from src.models.predictor import ModelPredictor


def test_predictor_model_loads():
    predictor = ModelPredictor()
    assert predictor.model is not None


def test_predictor_returns_expected_schema():
    predictor = ModelPredictor()
    sample_input = {
        "Customer_Age": 30,
        "Gender": "M",
        "Annual_Income": 50000,
        "Credit_Score": 700,
        "Purchase_Category": "Electronics",
        "BNPL_Provider": "ProviderA",
        "Purchase_Amount": 150.0,
        "Device_Type": "Mobile",
        "Connection_Type": "WiFi",
        "Checkout_Time_Seconds": 45,
        "Browser": "Chrome"
    }

    result = predictor.predict(sample_input)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"credit_score", "probability_of_default", "status"}
    assert 300 <= result["credit_score"] <= 850
    assert 0.0 <= result["probability_of_default"] <= 1.0
