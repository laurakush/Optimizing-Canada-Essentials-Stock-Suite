if __name__ == '__main__':
    # import librabries and set the device
    import torch
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MaxAbsScaler
    from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
    from finrl.meta.preprocessor.preprocessors import GroupByScaler
    from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv
    from finrl.agents.portfolio_optimization.models import DRLAgent
    from finrl.agents.portfolio_optimization.architectures import EIIE
    import optuna

    # Set device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(device)

    #Optimizing a Canadian Essentials Portfolio
    #CNR.TO - Canadian National Railway Co
    #CU.TO - Canadian Utilities Ltd
    #RY.TO - RBC
    #BNS.TO - Bank of Nova Scotia
    #TD.TO - TD Bank
    #ENB.TO - Enbridge Inc
    #L.TO - Loblaws
    #H.TO - Hydro One
    #BCE.TO - Bell
    #RCI-B.TO - Rogers
    CUSTOM_STOCK_LIST = ['CNR.TO', 'CU.TO', 'RY.TO', 'BNS.TO', 'TD.TO', 'ENB.TO', 'L.TO',
                     'H.TO', 'BCE.TO', 'RCI-B.TO']
    NUM_STOCKS = 10


    # Download stock data
    START_DATE = '2015-01-01'
    END_DATE = '2025-01-01'
    portfolio_raw_df = YahooDownloader(start_date=START_DATE,
                                   end_date=END_DATE,
                                   ticker_list=CUSTOM_STOCK_LIST).fetch_data()

    # Group by ticker and count occurrences


    # Normalize the data
    # You can use GroupByScaler with a MaxAbsScaler here
    portfolio_norm_df = GroupByScaler(scaler=MaxAbsScaler,
                                  by='tic').fit_transform(portfolio_raw_df)

    df_portfolio = portfolio_norm_df[["date", "tic", "close", "high", "low"]]
    ticker_counts = df_portfolio.groupby('date')['tic'].nunique()
    valid_dates = ticker_counts[ticker_counts == NUM_STOCKS].index
    df_portfolio = df_portfolio[df_portfolio['date'].isin(valid_dates)]

    # Split data into training and testing sets
    START_DATE_TRAIN = "2015-01-01"
    END_DATE_TRAIN = "2022-12-31"
    START_DATE_TEST = "2023-01-01"
    END_DATE_TEST = "2025-01-01"

    # Define your train and test data
    df_portfolio_train = df_portfolio[
        (df_portfolio["date"] >= START_DATE_TRAIN) & (df_portfolio["date"] < END_DATE_TRAIN)
        ].reset_index(drop=True)
    df_portfolio_test = df_portfolio[
        (df_portfolio["date"] >= START_DATE_TEST) & (df_portfolio["date"] < END_DATE_TEST)
        ].reset_index(drop=True)

    # print the train and test dfs shape
    TRAIN_DF_SHAPE = df_portfolio_train.shape
    TEST_DF_SHAPE = df_portfolio_test.shape
    print("Train df shape: ", TRAIN_DF_SHAPE)
    print("Test df shape: ", TEST_DF_SHAPE)

    # Define the environment
    # We will use portfolio optimization for the project
    INITIAL_AMOUNT = 1000000# initial amount of money in the portfolio: float
    COMISSION_FEE_PTC = 0.001# commission fee: float
    TIME_WINDOW = 30 # time window: int
    FEATURES = ["close", "high", "low"] # Market features used in training

    environment_train = PortfolioOptimizationEnv(
    df_portfolio_train,
    initial_amount=INITIAL_AMOUNT,
    comission_fee_pct=COMISSION_FEE_PTC,
    time_window=TIME_WINDOW,
    features=FEATURES,
    normalize_df=None, # df is already normalized
    order_df = False
    )

    enviroment_test = PortfolioOptimizationEnv(
        df_portfolio_test,
        initial_amount=INITIAL_AMOUNT,
        comission_fee_pct=COMISSION_FEE_PTC,
        time_window=TIME_WINDOW,
        features=FEATURES,
        normalize_df=None, # df is already normalized
        order_df = False
    )



    # Set PolicyGradient parameters
    # Set the learning rate for the training
    model_kwargs = {
    "lr": 0.001, #learning rate
    "policy": EIIE, #EIIE policy (Efficient Indepenent Investor Embedding)
    }

    # Set EIIE's parameters
    policy_kwargs = {
    "k_size": 3, #k_size: int (number of kernels in CNN [Convolutional Neural Network] layers)
    "time_window": TIME_WINDOW, # time window defined previously
    }

    # Instantiate the model
    model = DRLAgent(environment_train).get_model("pg", device, model_kwargs, policy_kwargs)

    # Train the model
    EPISODES = 500 # number of episodes to training the model: int

    DRLAgent.train_model(model, episodes=EPISODES)

    # Save the model
    torch.save(model.train_policy.state_dict(), "policy_EIIE.pt")

    # print the final value of the portfolio
    final_portfolio_value_train = environment_train._asset_memory["final"][-1]
    print("The final portfolio value at train is:", final_portfolio_value_train)


    EIIE_results = {
        "train": environment_train._asset_memory["final"],
        "test": {},
    }

    # instantiate an architecture with the same arguments used in training
    # and load with load_state_dict.
    policy = EIIE(time_window=TIME_WINDOW, device=device)
    policy.load_state_dict(torch.load("policy_EIIE.pt"))

    # testing
    DRLAgent.DRL_validation(model, enviroment_test, policy=policy)
    EIIE_results["test"] = enviroment_test._asset_memory["final"]

    # print the final value of the portfolio
    final_portfolio_value_test = enviroment_test._asset_memory["final"][-1]
    print("The final portfolio value at test is:", final_portfolio_value_test)

    # Define the objective function
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        k_size = trial.suggest_int("k_size", 1, 10)
        time_window =  trial.suggest_int("time_window", 10, 60)

        # set up train environment
        environment_train = PortfolioOptimizationEnv(
            df_portfolio_train,
            initial_amount=INITIAL_AMOUNT,
            comission_fee_pct=COMISSION_FEE_PTC,
            time_window=time_window,
            features=FEATURES,
            normalize_df=None,  # df is already normalized
            order_df=False
        )

        # setup model kwargs
        model_kwargs = {
            "lr": lr,
            "policy": EIIE,
        }
        # setup policy kwargs
        policy_kwargs = {
            "k_size": k_size,
            "time_window": time_window,
        }
        # Train model using DRLAgent
        model = DRLAgent(environment_train).get_model("pg", device, model_kwargs, policy_kwargs)
        DRLAgent.train_model(model, episodes=500)

        # define test environment
        # set up train environment
        environment_test = PortfolioOptimizationEnv(
            df_portfolio_test,
            initial_amount=INITIAL_AMOUNT,
            comission_fee_pct=COMISSION_FEE_PTC,
            time_window=time_window,
            features=FEATURES,
            normalize_df=None,  # df is already normalized
            order_df=False
        )

        # validate with test environment
        DRLAgent.DRL_validation(model, enviroment_test, policy=policy) # complete the code

        # final portfolio value as metric for the optimization
        return environment_test._asset_memory["final"][-1]


    # Create a study and optimize the objective function
    study = optuna.create_study(direction='maximize')
    N_TRIALS = 10
    study.optimize(objective, n_trials=N_TRIALS)

    # print the best hyperparameters
    BEST_HYPERPARAMETERS = study.best_params # using the study get the best hyperparameters
    print("Best hyperparameters: ", BEST_HYPERPARAMETERS)