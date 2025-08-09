import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def svm_regression(data_path, 
                   target_columns, 
                   test_size=0.2, 
                   random_state=42, 
                   kernel='rbf', 
                   C=1.0, 
                   epsilon=0.1,
                   plot_results=True):
    
    df = pd.read_excel(data_path, na_values='n.a.')
    
    df = df.dropna()
    
    X = df.drop(columns=target_columns).apply(pd.to_numeric, errors='coerce')
    y = df[target_columns].apply(pd.to_numeric, errors='coerce')
    
    valid_idx = ~X.isnull().any(axis=1) & ~y.isnull().any(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    results = {}
    
    for target in target_columns:
        model = SVR(kernel=kernel, C=C, epsilon=epsilon)
        model.fit(X_train_scaled, y_train[target].values)
        
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        train_mse = mean_squared_error(y_train[target], y_train_pred)
        train_r2 = r2_score(y_train[target], y_train_pred)
        
        test_mse = mean_squared_error(y_test[target], y_test_pred)
        test_r2 = r2_score(y_test[target], y_test_pred)
        
        results[target] = {
            'train_mse': train_mse,
            'train_r2': train_r2,
            'test_mse': test_mse,
            'test_r2': test_r2,
            'y_train': y_train[target].values,
            'y_train_pred': y_train_pred,
            'y_test': y_test[target].values,
            'y_test_pred': y_test_pred
        }
        models[target] = model
        
        if kernel == 'linear':
            coef_df = pd.DataFrame({
                'Feature': X.columns,
                'Coefficient': model.coef_[0]
            })
            results[target]['coef_df'] = coef_df
            
        if plot_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            ax1.scatter(y_train[target], y_train_pred, s=2, alpha=0.6, color='blue', label='Train Data')

            min_val_train = min(y_train[target].min(), y_train_pred.min())
            max_val_train = max(y_train[target].max(), y_train_pred.max())
            lims_train = [min_val_train - 0.1, max_val_train + 0.1]
            
            ax1.scatter(lims_train, lims_train, s=2, alpha=0.75, color='blue', label='Ideal Prediction')

            z_train = np.polyfit(y_train[target], y_train_pred, 1)
            p_train = np.poly1d(z_train)
            ax1.plot(y_train[target], p_train(y_train[target]), 'r-', label='Regression Line')
            
            ax1.set_xlabel('Actual Values')
            ax1.set_ylabel('Predicted Values')
            ax1.set_title(f'{target.upper()}: Train Set (n={len(y_train)}, MSE={train_mse:.2f}, R2={train_r2:.2f})')
            ax1.legend()
            ax1.grid(True)
            ax1.set_xlim(lims_train)
            ax1.set_ylim(lims_train)
            
            ax2.scatter(y_test[target], y_test_pred, s=2, alpha=0.6, color='orange', label='Test Data')
            
            min_val_test = min(y_test[target].min(), y_test_pred.min())
            max_val_test = max(y_test[target].max(), y_test_pred.max())
            lims_test = [min_val_test - 0.1, max_val_test + 0.1]
            
            ax2.plot(lims_test, lims_test, 'k--', alpha=0.75, label='Ideal Prediction')
            
            z_test = np.polyfit(y_test[target], y_test_pred, 1)
            p_test = np.poly1d(z_test)
            ax2.plot(y_test[target], p_test(y_test[target]), 'r-', label='Regression Line')
            
            ax2.set_xlabel('Actual Values')
            ax2.set_ylabel('Predicted Values')
            ax2.set_title(f'{target.upper()}: Test Set (n={len(y_test)}, MSE={test_mse:.2f}, R2={test_r2:.2f})')
            ax2.legend()
            ax2.grid(True)
            ax2.set_xlim(lims_test)
            ax2.set_ylim(lims_test)
            
            plt.tight_layout()
            plt.show()
    
    return models, results


if __name__ == "__main__":
    DATA_FILE = 'C:/Users/e/Desktop/Germany/learning/Data/house_data_2022.xlsx'  
    TARGET_COLUMNS = ['medv']
    
    models, results = svm_regression(
        data_path=DATA_FILE,
        target_columns=TARGET_COLUMNS, 
        kernel='rbf',       
        C=10.0,            
        epsilon=0.01,
        plot_results=True  
    )
    
    for target, result in results.items():
        print(f"\nEvaluation of target variable '{target.upper()}':")
        print(f"Train MSE: {result['train_mse']:.4f} (n={len(result['y_train'])})")
        print(f"Train R2: {result['train_r2']:.4f}")
        print(f"Test MSE: {result['test_mse']:.4f} (n={len(result['y_test'])})")
        print(f"Test R2: {result['test_r2']:.4f}")
        
        if 'coef_df' in result:
            print(f"\nFeature importance for target variable '{target.upper()}' (linear kernel):")
            print(result['coef_df'].sort_values('Coefficient', ascending=False))