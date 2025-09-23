from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.metrics import mean_squared_error

class CarpetBaggingRegressor:
    def __init__(self, estimator=None, n_estimators=100, random_state=None):
        """A Bagging Regressor with pruning and growth capabilities."""
        self.estimator = estimator or DecisionTreeRegressor()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = self._construct_model()

    def _construct_model(self):
        return BaggingRegressor(
            estimator=self.estimator,
            n_estimators=self.n_estimators,
            bootstrap=True,
            oob_score=True,
            random_state=self.random_state,
        )

    def removed_data_indices(self, y_big, y_small):
        mask = np.isin(y_big, y_small)
        return np.where(~mask)[0], np.where(mask)[0]

    def get_stats_full_model(self, X, y):
        y_pred = self.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        return np.mean(np.abs(y_pred - y)), np.sum((y_pred - y) ** 2), mse, np.sqrt(mse)

    def prune_my_data(self, X, y, num_to_prune, highlow, lst):
        indices_to_keep = np.sort(lst[: len(y) - num_to_prune]) if highlow == 1 else np.sort(lst[num_to_prune:])
        return X[indices_to_keep], y[indices_to_keep]

    def one_round_of_prune_and_train(self, X, y, num_to_prune):
        self.model.fit(X, y.ravel())  # Ensuring y is 1D
        y_pred = np.array([est.predict(X) for est in self.model.estimators_])
        var_tot = np.var(y_pred, axis=0)
        sorted_indices = np.argsort(var_tot)
        X_pruned, y_pruned = self.prune_my_data(X, y, num_to_prune, 0, sorted_indices)
        self.model.fit(X_pruned, y_pruned.ravel())  
        return self.get_stats_full_model(X, y), X_pruned, y_pruned



    def down_selection(self, X, y, prune_itt=10, prune_amt=100):
        """Iteratively prunes data and retrains the model, tracking performance."""
        err_counter = np.zeros(prune_itt + 1)
    
        self.model.fit(X, y.ravel())
        mae, ssqe, mse, rmse = self.get_stats_full_model(X, y)
        print(f"Initial RMSE on full dataset: {rmse}")
        err_counter[0] = rmse
    
        X_current, y_current = X, y
    
        for i in range(prune_itt):

            # Generate predictions ONCE per iteration
            y_pred = np.stack([est.predict(X_current) for est in self.model.estimators_])  
            var_tot = np.var(y_pred, axis=0)
            sorted_indices = np.argsort(var_tot)
    
            # Prune data
            X_new, y_new = self.prune_my_data(X_current, y_current, prune_amt, 0, sorted_indices)
    
            # Instead of retraining everything, refit on fewer trees
            self.model = BaggingRegressor(
                estimator=self.estimator,
                n_estimators=len(self.model.estimators_),  # Keep same number of trees
                bootstrap=True,
                oob_score=True,
                random_state=self.random_state,
            )
            self.model.fit(X_new, y_new.ravel())
    
            mae, ssqe, mse, rmse = self.get_stats_full_model(X, y)
            print(f"RMSE after {i + 1} round(s) of data down-selection: {rmse}")
    
            err_counter[i + 1] = rmse
            X_current, y_current = X_new, y_new
    
        return X_current, y_current, err_counter


    def up_selection(self, X, y, X_all, y_all, num_grows, growth_size, selector):
        err_counter = np.zeros((num_grows + 1, 2))
        add_pts = np.zeros((growth_size, num_grows), dtype=int)  

        self.model.fit(X, y.ravel())  
        err_counter[0] = self.get_stats_full_model(X_all, y_all)[:2]

        idx_in, idx_out = self.removed_data_indices(y_all, y)

        for i in range(num_grows):
            new_X, new_y, new_idx_in, new_idx_out = self.one_grow_cycle(X_all, y_all, idx_in, idx_out, growth_size, selector)
            
            self.model.fit(new_X, new_y.ravel())  

            err_counter[i + 1] = self.get_stats_full_model(X_all, y_all)[:2]
            print(f"RMSE after {i + 1} round(s) of data up-selection: {err_counter[i + 1, 0]}")
            print(f"Size of the dataset after round {i + 1}: {new_y.shape}")

            add_pts[:, i] = np.setdiff1d(new_idx_in, idx_in)

            idx_in, idx_out = new_idx_in, new_idx_out

        return err_counter, add_pts

    def one_grow_cycle(self, X_all, y_all, idx_in, idx_out, to_grow, selector):
        X_choice = X_all[idx_out]
        y_pred = np.array([est.predict(X_choice) for est in self.model.estimators_])

        var_tot = np.var(y_pred, axis=0)
        sorted_indices = np.argsort(var_tot)

        new_idx_in = np.sort(np.concatenate((idx_in, idx_out[sorted_indices[-to_grow:]])))
        new_idx_out = np.setdiff1d(idx_out, new_idx_in)

        return X_all[new_idx_in], y_all[new_idx_in], new_idx_in, new_idx_out
