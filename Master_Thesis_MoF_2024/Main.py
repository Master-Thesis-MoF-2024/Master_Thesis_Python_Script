"""
Applications of AI & Machine Learning in the Value Detection Process

Python script used for the analysis.
"""
# Importing External Libraries
import pandas as pd
import warnings

# Settings
warnings.simplefilter(action='ignore', category=Warning)
pd.set_option('display.max_columns', 40)

# Importing Internal Libraries & datasets
from data.loaded_datasets import df_2005, df_2006, df_2007, df_2008, \
    df_2009, df_2010, df_2011, df_2012, df_2013, df_2014, df_2015, df_2016,\
        df_2017, df_2018, df_2019, df_2020, df_2021, df_2022, df_2023, df_rtn, \
            df_benchmark, df_five_star, df_four_star
from Data_Preparation.Data_Preparation import Data_Preparation
from Data_Preparation.methods.calculate_outperformers import\
    calculate_outperformers
from Data_Preparation.methods.create_data_x_training import\
    create_data_x_training
from Training.Training import Training
from Model_Evaluation.Model_Evaluation import Model_Evaluation
from Performance_Simulation.Performance_Simulation  import Performance_Simulation
from Performance_Evaluation.methods.calculate_sharpe_ratio \
    import calculate_sharpe_ratio
from Performance_Evaluation.methods.calculate_max_drawdown \
    import calculate_max_drawdown
from Performance_Evaluation.methods.calculate_information_ratio \
    import calculate_information_ratio
from Performance_Evaluation.methods.calculate_jensen_alpha \
    import calculate_jensen_alpha
from Performance_Evaluation.methods.plot_performances \
    import plot_performances   
from Hypothesis_Testing.Hypothesis_Testing import Hypothesis_Testing

"""
Chapter I: Preparing Data for analysis
"""
# Initializing class dedicated to data preparation and cleaning all datasets
# by removing outliers and by substituting missing values with the average 
# for the year, plus we calculate if the stock has been an outperformer in the relevant period 
df_2005 = calculate_outperformers((Data_Preparation(df_2005).clean_data()), \
                                 "Log_rtn")
df_2006 = calculate_outperformers((Data_Preparation(df_2006).clean_data()), \
                                 "Log_rtn")
df_2007 = calculate_outperformers((Data_Preparation(df_2007).clean_data()), \
                                 "Log_rtn")
df_2008 = calculate_outperformers((Data_Preparation(df_2008).clean_data()), \
                                 "Log_rtn")
df_2009 = calculate_outperformers((Data_Preparation(df_2009).clean_data()), \
                                 "Log_rtn")
df_2010 = calculate_outperformers((Data_Preparation(df_2010).clean_data()), \
                                 "Log_rtn")
df_2011 = calculate_outperformers((Data_Preparation(df_2011).clean_data()), \
                                 "Log_rtn")
df_2012 = calculate_outperformers((Data_Preparation(df_2012).clean_data()), \
                                 "Log_rtn")
df_2013 = calculate_outperformers((Data_Preparation(df_2013).clean_data()), \
                                 "Log_rtn")
df_2014 = calculate_outperformers((Data_Preparation(df_2014).clean_data()), \
                                 "Log_rtn")
df_2015 = calculate_outperformers((Data_Preparation(df_2015).clean_data()), \
                                 "Log_rtn")
df_2016 = calculate_outperformers((Data_Preparation(df_2016).clean_data()), \
                                 "Log_rtn")
df_2017 = calculate_outperformers((Data_Preparation(df_2017).clean_data()), \
                                 "Log_rtn")
df_2018 = calculate_outperformers((Data_Preparation(df_2018).clean_data()), \
                                 "Log_rtn")
df_2019 = calculate_outperformers((Data_Preparation(df_2019).clean_data()), \
                                 "Log_rtn")
df_2020 = calculate_outperformers((Data_Preparation(df_2020).clean_data()), \
                                 "Log_rtn")
df_2021 = calculate_outperformers((Data_Preparation(df_2021).clean_data()), \
                                 "Log_rtn")
df_2022 = calculate_outperformers((Data_Preparation(df_2022).clean_data()), \
                                 "Log_rtn")
df_2023 = calculate_outperformers((Data_Preparation(df_2023).clean_data()), \
                                 "Log_rtn")

# Preparing datasets to train ML algorithms
X_2005, y_2005 = create_data_x_training(df_2005, "Outperform")
X_2006, y_2006 = create_data_x_training(df_2006, "Outperform")
X_2007, y_2007 = create_data_x_training(df_2007, "Outperform")
X_2008, y_2008 = create_data_x_training(df_2008, "Outperform")
X_2009, y_2009 = create_data_x_training(df_2009, "Outperform")
X_2010, y_2010 = create_data_x_training(df_2010, "Outperform")
X_2011, y_2011 = create_data_x_training(df_2011, "Outperform")
X_2012, y_2012 = create_data_x_training(df_2012, "Outperform")
X_2013, y_2013 = create_data_x_training(df_2013, "Outperform")
X_2014, y_2014 = create_data_x_training(df_2014, "Outperform")
X_2015, y_2015 = create_data_x_training(df_2015, "Outperform")
X_2016, y_2016 = create_data_x_training(df_2016, "Outperform")
X_2017, y_2017 = create_data_x_training(df_2017, "Outperform")
X_2018, y_2018 = create_data_x_training(df_2018, "Outperform")
X_2019, y_2019 = create_data_x_training(df_2019, "Outperform")
X_2020, y_2020 = create_data_x_training(df_2020, "Outperform")
X_2021, y_2021 = create_data_x_training(df_2021, "Outperform")
X_2022, y_2022 = create_data_x_training(df_2022, "Outperform")
X_2023, y_2023 = create_data_x_training(df_2023, "Outperform")

"""
Chapter II: Training the Models
"""
# Training relevant ML algorithms to understand which one is the best performing one
# !!Important!!: Hypeparamther Optimization already conducted during the training process

# As anticipated in the Thesis document training will be from 2005 to 2017
X_train = pd.concat([X_2005,X_2006,X_2007,X_2008,X_2009,X_2010,X_2011,\
                     X_2012,X_2013,X_2014,X_2015,X_2016,X_2017],axis=0)
y_train = pd.concat([y_2005,y_2006,y_2007,y_2008,y_2009,y_2010,y_2011,\
                     y_2012,y_2013,y_2014,y_2015,y_2016,y_2017],axis=0)
    
# While testing will be from 2018 to 2023
X_test = pd.concat([X_2018,X_2019,X_2020,X_2021,X_2022,X_2023],axis=0)
y_test = pd.concat([y_2018,y_2019,y_2020,y_2021,y_2022,y_2023],axis=0)

#Training & Tuning models
svm = Training.train_svm(X_train, y_train) # Support Vector Classifier
random_forest = Training.train_random_forest(X_train, y_train) # Random Forest
logistic_reg = Training.train_logistic_regression(X_train, y_train) # Logistic Reg
knn = Training.train_knn(X_train, y_train) # K-Nearest Neighbours 
gaussia_nb = Training.train_gaussian_nb_c(X_train, y_train) # Gaussian Naive Bayes
mlp = Training.train_NN(X_train, y_train) # DL MLP


"""
Chapter III: Evaluating model performances
"""
# Predicting over the test period with the trained models
y_pred_svm = svm.predict(X_test)
y_pred_rf = random_forest.predict(X_test)
y_pred_lg = logistic_reg.predict(X_test)
y_pred_knn = knn.predict(X_test)
y_pred_gnb = gaussia_nb.predict(X_test)
y_pred_mlp = mlp.predict(X_test)

# Initializing Evaluation Module
svm_ev = Model_Evaluation(svm)
random_forest_ev = Model_Evaluation(random_forest)
logistic_reg_ev = Model_Evaluation(logistic_reg)
knn_ev = Model_Evaluation(knn)
gaussia_nb_ev = Model_Evaluation(gaussia_nb)
mlp_ev = Model_Evaluation(mlp)

# Calculating selected performance metrics for each algorithm
# We will use: Accuracy, ROC-AUC, Precision-Recall-F1, Confusion Matrix

# Calculating Accuracy 
ac_svm = svm_ev.calculate_accuracy(y_test, y_pred_svm)
ac_rf = random_forest_ev.calculate_accuracy(y_test, y_pred_rf)
ac_lg = logistic_reg_ev.calculate_accuracy(y_test, y_pred_lg)
ac_knn = knn_ev.calculate_accuracy(y_test, y_pred_knn)
ac_gnb = gaussia_nb_ev.calculate_accuracy(y_test, y_pred_gnb)
ac_mlp = mlp_ev.calculate_accuracy(y_test, y_pred_mlp)

# Displaying results
print(f"Accuracy SVM: {ac_svm:.4f}")
print(f"Accuracy Random Forest: {ac_rf:.4f}")
print(f"Accuracy Logistic Regression: {ac_lg:.4f}")
print(f"Accuracy KNN: {ac_knn:.4f}")
print(f"Accuracy Gaussian NB: {ac_gnb:.4f}")
print(f"Accuracy MLP: {ac_mlp:.4f}")

# Calculating Precision, Recall & F1 Report 
ac_svm = svm_ev.calculate_precision_recall_f1(y_test, y_pred_svm)
ac_rf = random_forest_ev.calculate_precision_recall_f1(y_test, y_pred_rf)
ac_lg = logistic_reg_ev.calculate_precision_recall_f1(y_test, y_pred_lg)
ac_knn = knn_ev.calculate_precision_recall_f1(y_test, y_pred_knn)
ac_gnb = gaussia_nb_ev.calculate_precision_recall_f1(y_test, y_pred_gnb)
ac_mlp = mlp_ev.calculate_precision_recall_f1(y_test, y_pred_mlp)

# Plotting Confusion Matrix
svm_ev.calculate_confusion_matrix(y_test, y_pred_svm, "SVC")
random_forest_ev.calculate_confusion_matrix(y_test, y_pred_rf, \
                                            "Random Forest")
logistic_reg_ev.calculate_confusion_matrix(y_test, y_pred_lg, \
                                           "Logistic Regression")
knn_ev.calculate_confusion_matrix(y_test, y_pred_knn, "KNN")
gaussia_nb_ev.calculate_confusion_matrix(y_test, y_pred_gnb, \
                                         "Gaussian Naive Bayes")
mlp_ev.calculate_confusion_matrix(y_test, y_pred_mlp, "MLP")

# Plotting ROC-AUC Curve
svm_ev.plot_roc_auc(y_test, y_pred_svm, "SVC")
random_forest_ev.plot_roc_auc(y_test, y_pred_rf, "Random Forest")
logistic_reg_ev.plot_roc_auc(y_test, y_pred_lg, "Logistic Regression")
knn_ev.plot_roc_auc(y_test, y_pred_knn, "KNN")
gaussia_nb_ev.plot_roc_auc(y_test, y_pred_gnb, "Gaussian Naive Bayes")
mlp_ev.plot_roc_auc(y_test, y_pred_mlp, "MLP")

# At the end of the evaluation process, it was deducted that the best performing 
# model for this kind of task is "Random Forest"
# Thus, it will now be integrated in the performance simulation of the portfolios

"""
Chapter IV: Performance Simulation
"""
# Grouping under one variable all the datasets that will be implemented in the
# simulation

# Grouping Train Sample
rolling_X_train_sample = [X_2005, X_2006, X_2007, X_2008, X_2009, X_2010, X_2011,\
                          X_2012, X_2013, X_2014, X_2015, X_2016, X_2017, X_2018,\
                              X_2019, X_2020, X_2021, X_2022]
rolling_y_train_sample = [y_2005, y_2006, y_2007, y_2008, y_2009, y_2010, \
                          y_2011, y_2012, y_2013, y_2014, y_2015, y_2016, y_2017,\
                              y_2018, y_2019, y_2020, y_2021, y_2022]

# Grouping Test Sample
rolling_monthly_test_sample = [df_2006, df_2007, df_2008, df_2009, df_2010, df_2011,\
                               df_2012, df_2013, df_2014, df_2015, df_2016, df_2017,\
                                   df_2018, df_2019, df_2020, df_2021, df_2022, df_2023]

# Initializing Simulation Module
simulator = Performance_Simulation()

# Simulating Trading Strategy (No trading costs)
simulated_portfolio = simulator.simulate_yearly_rebalancing(rolling_X_train_sample,\
                                                            rolling_y_train_sample,\
                                                                rolling_monthly_test_sample,\
                                                                    df_rtn) # df_rtn contains the monthly returns
                                                                            # of all the stocks in the index 
                                                                            # from 2005 to 2023

"""
Chapter V: Performance Comparison 
"""

"""Adjusting samples for comparison with Alken Fund - Small Cap Europe Class EU1"""
# Returns are calculated from Oct-2010 to Dec-2023

simulated_portfolio_for_comp_5s = simulated_portfolio[41:]

index_for_comp_5s = df_benchmark[41:]

"""Adjusting samples for comparison with Danske Invest SICAV - Europe Small Cap I""" 
# Returns are calculated from Sep-2013 to Dec-2023

simulated_portfolio_for_comp_4s = simulated_portfolio[76:]
index_for_comp_4s = df_benchmark[76:]


"""Portfolio vs Benchmark"""
# Plotting them
plot_performances(simulated_portfolio, df_benchmark, ["Simulated Portfolio",\
                                                        "EURO STOXX"])
print("Portfolio vs Index")
# Calculating Sharpe ratio for both of them
sharpe_portfolio = calculate_sharpe_ratio(simulated_portfolio)
sharpe_index = calculate_sharpe_ratio(df_benchmark)

print(f"\nSharpe for Simulated Portfolio: {sharpe_portfolio}")
print(f"Sharpe for Euro Stoxx: {sharpe_index}")

# Calculating Max Drawdown for both of them
max_draw_portfolio = calculate_max_drawdown(simulated_portfolio)
max_draw_index = calculate_max_drawdown(df_benchmark)

print(f"Max Drawdown for Simulated Portfolio: {max_draw_portfolio}")
print(f"Max Drawdown for Euro Stoxx: {max_draw_index}")

# Calculating Jensen alpha for portfolio
alpha_portfolio = calculate_jensen_alpha(simulated_portfolio, df_benchmark)

print(f"Alpha for Simulated Portfolio: {alpha_portfolio}")

# Calculating Information Ratio for portfolio
information_ratio_portfolio = calculate_information_ratio(simulated_portfolio, df_benchmark)

print(f"Information ratio for Simulated Portfolio: {information_ratio_portfolio}")


"""Portfolio vs 5-star fund"""
# Plotting them
plot_performances(simulated_portfolio_for_comp_5s, df_five_star, ["Simulated Portfolio",\
                                                        "Alken Fund"])
print("Portfolio vs 5-star fund")
# Calculating Sharpe ratio for all of them
sharpe_portfolio_comp5 = calculate_sharpe_ratio(simulated_portfolio_for_comp_5s)
sharpe_5_star = calculate_sharpe_ratio(df_five_star)
sharpe_index_comp5 = calculate_sharpe_ratio(index_for_comp_5s)

print(f"\nSharpe for Simulated Portfolio: {sharpe_portfolio_comp5}")
print(f"Sharpe for Euro Stoxx: {sharpe_index_comp5}")
print(f"Sharpe for Alken Fund: {sharpe_5_star}")

# Calculating Max Drawdown for all of them
max_draw_portfolio_comp5 = calculate_max_drawdown(simulated_portfolio_for_comp_5s)
max_draw_5_star = calculate_max_drawdown(df_five_star)
max_draw_index_comp5 = calculate_max_drawdown(index_for_comp_5s)

print(f"Max Drawdown for Simulated Portfolio: {max_draw_portfolio_comp5}")
print(f"Max Drawdown for Euro Stoxx: {max_draw_index_comp5}")
print(f"Max Drawdown for Alken Fund: {max_draw_5_star}")

# Calculating Jensen alpha for all of them
alpha_portfolio_comp5 = calculate_jensen_alpha(simulated_portfolio_for_comp_5s,\
                                                  index_for_comp_5s)
alpha_5_star = calculate_jensen_alpha(df_five_star, index_for_comp_5s)

print(f"Alpha for Simulated Portfolio: {alpha_portfolio_comp5}")
print(f"Alpha for Alken Fund: {alpha_5_star}")

# Calculating information ratio for all of them
information_ratio_portfolio_comp5 = calculate_information_ratio(simulated_portfolio_for_comp_5s,\
                                                                index_for_comp_5s)
information_ratio_5_star = calculate_information_ratio(df_five_star, index_for_comp_5s)

print(f"Information ratio for Simulated Portfolio: {information_ratio_portfolio_comp5}")
print(f"Information ratio for Alken Fund: {information_ratio_5_star}")

"""Portfolio vs 4-star fund"""
# Plotting them
plot_performances(simulated_portfolio_for_comp_4s, df_four_star, ["Simulated Portfolio",\
                                                        "Danske Fund"])

print("Portfolio vs 4-star fund")
# Calculating Sharpe ratio for all of them
sharpe_portfolio_comp4 = calculate_sharpe_ratio(simulated_portfolio_for_comp_4s)
sharpe_4_star = calculate_sharpe_ratio(df_four_star)
sharpe_index_comp4 = calculate_sharpe_ratio(index_for_comp_4s)

print(f"\nSharpe for Simulated Portfolio: {sharpe_portfolio_comp4}")
print(f"Sharpe for Euro Stoxx: {sharpe_index_comp4}")
print(f"Sharpe for Danske Fund: {sharpe_4_star}")

# Calculating Max Drawdown for all of them
max_draw_portfolio_comp4 = calculate_max_drawdown(simulated_portfolio_for_comp_4s)
max_draw_4_star = calculate_max_drawdown(df_four_star)
max_draw_index_comp4 = calculate_max_drawdown(index_for_comp_4s)

print(f"Max Drawdown for Simulated Portfolio: {max_draw_portfolio_comp4}")
print(f"Max Drawdown for Euro Stoxx: {max_draw_index_comp4}")
print(f"Max Drawdown for Danske Fund: {max_draw_4_star}")

# Calculating Jensen alpha for all of them
alpha_portfolio_comp4 = calculate_jensen_alpha(simulated_portfolio_for_comp_4s,\
                                                  index_for_comp_4s)
alpha_4_star = calculate_jensen_alpha(df_four_star, index_for_comp_4s)

print(f"Alpha for Simulated Portfolio: {alpha_portfolio_comp4}")
print(f"Alpha for Danske Fund: {alpha_4_star}")

# Calculating information ratio for all of them
information_ratio_portfolio_comp4 = calculate_information_ratio(simulated_portfolio_for_comp_4s,\
                                                                index_for_comp_4s)
information_ratio_4_star = calculate_information_ratio(df_four_star, index_for_comp_4s)

print(f"Information ratio for Simulated Portfolio: {information_ratio_portfolio_comp4}")
print(f"Information ratio for Danske Fund: {information_ratio_4_star}")

"""
Chapter VI: Hypothesis Testing
"""
# Initializing Testing Module
tester = Hypothesis_Testing()

# H0: Machine Learning is not capable of outperforming the investment universe from which it draws its investment decisions.
# H1: Machine Learning is capable of outperforming the investment universe from which it draws its investment decisions.
print("\nTesting --> H0: Machine Learning is not capable of outperforming the investment universe from which it draws its investment decisions")
tester.test_Ho(simulated_portfolio, df_benchmark)

# H0: Machine Learning is not capable of outperforming the investment strategies actively managed by successful fund managers
# H1: Machine Learning is capable of outperforming the investment strategies actively managed by successful fund managers
print("\nTesting --> H0: Machine Learning is not capable of outperforming the investment strategies actively managed by successful fund managers")
print("Against Alken 5-star fund:")
tester.test_Ho(simulated_portfolio_for_comp_5s, df_five_star)

print("\nTesting --> H0: Machine Learning is not capable of outperforming the investment strategies actively managed by successful fund managers")
print("Against Danske 4-star fund:")
tester.test_Ho(simulated_portfolio_for_comp_4s, df_four_star)
