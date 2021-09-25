# JuliaPerturbation
Replication material package for "Perturbating and Estimating DSGE models in Julia" by Salazar-Perez and Seoane

July 2021

### Contents ###
	1. run_solution_xxxx: 	Launches the solution of a model.
	2. run_estimation_xxxx: Launches the Bayesian estimation of a model through the Sequential Monte Carlo (SMC) method.
	4. solution_functions: 	Repository with all the functions required to solve and simulate the model.	 	
	3. smc_rwmh_xxxx:	Repository with all the functions required to estimate the model through the SMC method. 

### Solution and simulation of the model ###
	In order to adapt the codes to solve and simulate a specific model the following file must be modified:
	
	1. run_solution_xxxx:
		## Model ##
    			# Settings #
        			flag_order = 		State the order of the estimation (1,2,3).
        			flag_deviation =	State whether the solution of the model should be log-linearized 
							(true) or not (false).

    			# Parameters
        			@vars 			List all the parameters in the model in order to generate them as symbols.
            			parameters = [;]	List all the parameters in the model in a column vector.
				estimate   = []		Since no parameter should be estimated, leave this field as an 
							empty vector.

    			# Variables
       				@vars 			List all the variables in the model in order to generate them as symbols.
           			x   =	[;]		List all the states in the model at time t as a column vector .
           			y   = 	[;]		List all the jumpers in the model at time t as a column vector. 
           			xp  = 	[;]		List all the states in the model at time t+1 as a column vector.
           			yp  = 	[;]		List all the jumpers in the model at time t+1 as a column vector.

   			# Shock
        			@vars 			List all the exogenous shocks in the model at time t in order to 
							generate them as symbols.
            			e   =   [;]		List as a column vector all the exogenous shocks affecting the 
							model at time t.
            			eta =   Array([])

    			# Equilibrium conditions
            			f1  =   		Write each of the equilibrium conditions as a symbolic function.
            			f2  =               
		  		... = 	

            			f   =   [f1;f2;f3]	List all the equilibrium conditions as a column vector. 

    			# Steady state (OPTIONAL)
    
            			SS = [;] 		If known, provide the symbolic expresion of the steady state in 
							terms of the parameters.
							It must be provided as a column vector, in which the variables 
							should be ordered in the following way: [x; y; xp; yp].
							If the steady state is unknown, leave it as an empty vector (SS=[]),
							so that the program tries to estimate it numerically.
		[...]
		## Solution ##
     			# Parametrization
        			PAR     =   [;]		List the value taken by all the parameters of the model as a column vector. 
							The order of the list should coincide with that of the previously declared
							"parameters" vector.
		[...]
		## Simulation ##
    			# Simulation
        			T_S 	= 		Indicate the number of periods to simulate.
        			IS_S 	= 		Indicate the initial state for the simulation. 
							To start in the staeady state state: 'zeros(model.nx)'
        	[...]
    			# Impulse-response functions
        			T_IR 	= 		Indicate the number of periods that the impulse response functions 
							should cover.
        			IS_IR 	= 		Indicate as a colomn vector the initial shocked state.

        	
## Estimation
	In order to adapt the codes of the estimation algorithm to a specific model the following files must be modified:
	
	2. run_estimation_xxxx:
		## Model
    			# Ajustes
        			flag_order = 		State the order of the estimation (1,2,3).
        			flag_deviation =	State whether the solution of the model should be log-linearized 
							(true) or not (false).

    			# Parameters
        			@vars 			List all the parameters in the model to generate them as symbols.
            			parameters = [;]	List all the parameters in the modelas a column vector.
				estimate   = [;]	List the parameters that need to be estimated in the model
							as a column vector. 

    			# Variables
       				@vars 			List all the variables in the model to generate them as symbols.
           			x   =	[;]		List all the states in the model at time t as a column vector.
           			y   = 	[;]		List all the jumpers in the model at time t as a column vector.
           			xp  = 	[;]		List all the states in the model at time t+1 as a column vector.
           			yp  = 	[;]		List all the jumpers in the model at time t+1 as a column vector.

   			# Shock
        			@vars 			List all the exogenous shocks in the model at time t in order to 
							generate them as symbols.
            			e   =   [;]		List as a column vector all the exogenous shocks affecting the 
							model at time t.
            			eta =   Array([])

    			# Equilibrium conditions
            			f1  =   		Write each of the equilibrium conditions as a symbolic function.
            			f2  =               
		  		... = 	

            			f   =   [f1;f2;f3]	List all the equilibrium conditions as a column vector . 

    			# Steady state (OPTIONAL)
    
            			SS = [;] 		If known, provide the symbolic expresion of the steady state in 
							terms of the parameters.
							It must be provided as a column vector, in which the variables 
							should be ordered in the following way: [x; y; xp; yp].
							If the steady state is unknown leave it as an empty vector (SS=[]), 
							so that the program tries to estimate it.
		[...]
		## Solution
     			# Parametrization
        			PAR     =   [;]		List the value taken by all the parameters in the model as a column vector. 
							The order should coincide with that of the previously created "parameters" 
							vector.
							An initial value for the parameters to be estimated must be included 
							in the vector.
		[...]
		## Estimation
        		initial_para 	=   [;]		List the initial values for the estimated parameters as a column vector.
			@time estimation_results = smc_run_rwmh_xxxx(initial_para, data', model, PAR)	
							This line initializes the SMC estimation method. 
							The data should be previously loaded in the variable 'data' 
							as column time series.

	4. smc_rwmh_xxxx:
		[...]
		line 55 	Prior distributions	Introduce the information of the prior distribution of every parameter 
							to be estimated, so that the vector "parasim" is filled with random 
							extractions from these distributions.				
