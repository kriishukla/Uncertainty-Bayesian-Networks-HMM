#############
## Imports ##
#############

import pickle
import pandas as pd
import numpy as np
import bnlearn as bn
from test_model import test_model

######################
## Boilerplate Code ##
######################

def load_data():
    """Load train and validation datasets from CSV files."""
    # Implement code to load CSV files into DataFrames
    # Example: train_data = pd.read_csv("train_data.csv")
    train_df = pd.read_csv('dataset csv files/train_data.csv')
    val_df = pd.read_csv('dataset csv files/validation_data.csv')

    Distance_label = {"short": 1, "medium": 2, "long": 3}
    Fare_Category_label = {"Low": 1, "Medium": 2, "High": 3}

    
    train_df['Distance'] = train_df['Distance'].map(Distance_label)
    train_df['Fare_Category'] = train_df['Fare_Category'].map(Fare_Category_label)

    val_df['Distance'] = val_df['Distance'].map(Distance_label)
    val_df['Fare_Category'] = val_df['Fare_Category'].map(Fare_Category_label)
    
    return train_df, val_df   
def make_network(df):
    """Define and fit the initial Bayesian Network."""
    # Code to define the DAG, create and fit Bayesian Network, and return the model
    features = df.columns.tolist()
    
    edges = [(features[i], features[j]) for i in range(len(features)) for j in range(i + 1, len(features))]
    
    dag = edges
    
    model = bn.make_DAG(dag)
    model = bn.parameter_learning.fit(model, df)
    
    bn.plot(model)
    
    return model

def make_pruned_network(df):
    """Define and fit a pruned Bayesian Network."""
    # Code to create a pruned network, fit it, and return the pruned model
    features = df.columns.tolist()
    
    edges = [(features[i], features[j]) for i in range(len(features)) for j in range(i + 1, len(features))]
    
    dag = edges
    pruned_dag = []
    edges_pruned = []

    corr_matrix = df.corr()
    index = 0
    
    while index < len(dag):
        edge = dag[index]
        node1, node2 = edge
        if node1 in corr_matrix.columns and node2 in corr_matrix.columns:
            corr_value = abs(corr_matrix.loc[node1, node2]) 
            (pruned_dag.append(edge) if corr_value >= 0.2 else edges_pruned.append(edge))
        index += 1
    
    print(f"Edges pruned: {edges_pruned}")
    
    model = bn.make_DAG(pruned_dag)
    model = bn.parameter_learning.fit(model, df)
    bn.plot(model)
    return model


def make_optimized_network(df):
    """Perform structure optimization and fit the optimized Bayesian Network."""
    # Code to optimize the structure, fit it, and return the optimized model
    features = df.columns.tolist()
    
    edges = [(features[i], features[j]) for i in range(len(features)) for j in range(i + 1, len(features))]
    
    dag = edges
    model = bn.structure_learning.fit(df, methodtype='hc',bw_list_method='edges',white_list=dag)
    model = bn.parameter_learning.fit(model, df)
    bn.plot(model)
    return model


def save_model(fname, model):
    """Save the model to a file using pickle."""
    with open(fname, 'wb') as f:
        pickle.dump(model, f)

def evaluate(model_name, val_df):
    """Load and evaluate the specified model."""
    with open(f"{model_name}.pkl", 'rb') as f:
        model = pickle.load(f)
        correct_predictions, total_cases, accuracy = test_model(model, val_df)
        print(f"Total Test Cases: {total_cases}")
        print(f"Total Correct Predictions: {correct_predictions} out of {total_cases}")
        print(f"Model accuracy on filtered test cases: {accuracy:.2f}%")

############
## Driver ##
############

def main():
    # Load data
    train_df, val_df = load_data()

    # Create and save base model
    base_model = make_network(train_df.copy())
    save_model("base_model.pkl", base_model)

    # Create and save pruned model
    pruned_network = make_pruned_network(train_df.copy())
    save_model("pruned_model.pkl", pruned_network)

    # Create and save optimized model
    optimized_network = make_optimized_network(train_df.copy())
    save_model("optimized_model.pkl", optimized_network)

    # Evaluate all models on the validation set
    evaluate("base_model", val_df)
    evaluate("pruned_model", val_df)
    evaluate("optimized_model", val_df)

    print("[+] Done")

if __name__ == "__main__":
    main()

