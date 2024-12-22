import pickle
import pandas as pd
import numpy as np
import bnlearn as bn
from sklearn.model_selection import train_test_split

def test_model(graph, val_df):
    defined_states = {}

    # Check if DAG has a model attribute and CPDs are available
    if 'model' in graph and hasattr(graph['model'], 'cpds'):
        for cpd in graph['model'].cpds:
            if hasattr(cpd, 'state_names') and cpd.variable in cpd.state_names:
                # Store the state names for each variable
                defined_states[cpd.variable] = cpd.state_names[cpd.variable]
    else:
        return 0, 0, 0  

    bn_variables = list(defined_states.keys())

    filtered_test_cases = val_df[bn_variables + ['Fare_Category']].to_dict(orient='records')

    correct_predictions = 0
    total_cases = len(filtered_test_cases)

    for evidence in filtered_test_cases:
        evidence_for_inference = {k: v for k, v in evidence.items() if k != 'Fare_Category'}

        try:
            inference_result = bn.inference.fit(graph, variables=['Fare_Category'], evidence=evidence_for_inference)
            probabilities = inference_result.values

            predicted_fare_category = inference_result.state_names['Fare_Category'][probabilities.argmax()]

            if predicted_fare_category == evidence['Fare_Category']:
                correct_predictions += 1

        except KeyError:
            pass  
        except Exception:
            pass  
        
    # Calculate accuracy
    accuracy = (correct_predictions / total_cases) * 100 if total_cases > 0 else 0

    return correct_predictions, total_cases, accuracy
