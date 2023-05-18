# DeRDaVa's User Guide

The objective of the ``derdava`` package is to perform data valuation in machine learning (ML), through which we know the value or worth of each data source. To start data valuation, we need to initialize the following:

* Data sources: Each data source can be either a single data point, or a smaller dataset. We require data sources to be a dictionary:
  
  ```
  data_sources = { 0: (X_0, y_0), 1: (X_1, y_1) }
  ```
  You can also generate random data sources from one of the built-in datasets (see ``derdava.dataset.load_dataset()``). For example,
  ```
  from derdava.data_source import generate_random_data_sources
  from derdava.dataset import load_dataset
  
  X, y = load_dataset('phoneme')
  data_sources = generate_random_data_sources(X, y, num_of_data_sources=10)
  ```
* ML model: You need to load built-in models:
  ```
  from derdava.model_utility import model_knn
  
  model = model_knn
  ```
  
Now we can start performing data valuation. Follow the steps below:

* Create a model utility function of class `ModelUtilityFunction`. For example,
  ```
  from derdava.model_utility import IClassificationModel
  
  model_utility_function = IClassificationModel(model, data_sources, X_test, y_test)
  ```
* If you are using DeRDaVa, you need to create a `CoalitionProbability` to tell you the staying probability of each coalition:
  ```
  from derdava.coalition_probability import IndependentCoalitionProbability
  
  staying_probabilities = { i: 0.5 for i in range(10) }
  coalition_probability = IndependentCoalitionProbability(staying_probabilities)
  ```
* We can finally do data valuation:
  ```
  from derdava.data_valuation import ValuableModel
  
  support = tuple(range(10))
  valuable_model = ValuableModel(support, model_utility_function)
  shapley_values = valuable_model.valuate(data_valuation_function='shapley')
  zot_mcmc_beta_16_1_values = valuable_model.valuate(data_valuation_function='012-mcmc robust beta', alpha=16, beta=4, tolerance=1.005)
  ```
  
Please refer to the documentation for more details on each submodules.
