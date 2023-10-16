#### Perturbating and Estimating DSGE Models in Julia ####

### Contents
	1. run_solution_xxxx: 	Lauches the solution of a model.
	2. run_estimation_xxxx: Launches the Bayesian estimation of a model through the Sequential Monte Carlo (SMC) method.
	4. solution_functions: 	Repository with all the functions required to solve and simulate the model.	 	

### Solution and simulation of the model
	The following file should be modified in order to adapt the codes to solve and simulate a specific model:	
	1. run_solution_xxxx:
		## Model
			# Ajustes
				flag_order 		= 	State the order of the estimation (1,2,3).
				flag_deviation 	=	State whether the solution of the model should be log-linearized 
									(true) or not (false).

			# Parameters
				@vars 				List all the parameters in the model to generate them as symbols.
				parameters 	= [;]	List in a column vector all the parameters in the model.
				estimate   	= []	Since no parameter should be estimated, leave this field as an 
									empty vector.
				position   	= []	Since no parameter should be estimated, leave this field as an 
									empty vector.
				priors   	= (;)	Since no parameter should be estimated, leave this field as an 
									empty vector.					

			# Variables
				@vars 				List all the variables in the model to generate them as symbols.
				x   =	[;]			List in a column vector all the states in the model at time t.
				y   = 	[;]			List in a column vector all the jumpers in the model at time t.
				xp  = 	[;]			List in a column vector all the states in the model at time t+1.
				yp  = 	[;]			List in a column vector all the jumpers in the model at time t+1.

			# Shock
				@vars 				List all the exogenous shocks affecting the model at time t to 
									generate them as symbols.
				e   = [;]			List in a column vector all the exogenous shocks affecting the 
									model at time t.
				eta = Array([])

			# Equilibrium conditions
				f1  =   			Write each of the equilibrium conditions as symbolic functions.
				f2  =               
				... = 	

				f   =   [f1;f2;f3]	List in a column vector all the equilibrium conditions. 

			# Steady state (OPTIONAL)
				SS = [;] 			If known, provide the symbolic expresion of the steady state in 
									terms of the parameters.
									It should be provided as a column vector, in which the variables 
									should be ordered in the following order: [x; y; xp; yp].
									If the steady state is unknown leave it as an empty vector (SS=[]),
									so that the program tries to estimate it.

		## Solution
			# Parametrization
				PAR = [;]			List in a column vector the specific value of all the parameters 
									in the model. 
									The order should coincide with that of the parameters vector.
			
		## Simulation
			# Simulation
        		flag_logdev = 		State whether the results should be build in terms of 
									log-deviation (true) or levels (false).
        		T_SM = 				Indicate the number of periods that the simulation 
									should cover.

    		# Impulse-response functions
        		flag_logdev = 		State whether the results should be build in terms of 
									log-deviation (true) or levels (false).
        		T_IR = 				Indicate the number of periods that the impulse response functions 
									should cover.
        	
## Estimation
	The following file should be modified in order to adapt the codes to estimate the parameters of a specific model:
	2. run_estimation_xxxx:
		## Model
			# Ajustes
				flag_order 		= 	State the order of the estimation (1,2,3).
				flag_deviation 	=	State whether the solution of the model should be log-linearized 
									(true) or not (false).

			# Parameters
				@vars 				List all the parameters in the model to generate them as symbols.
				parameters 	= [;]	List in a column vector all the parameters in the model.
				estimate   	= [;]	List in a column vector the parameters that need to be estimated 
									in the model.
				position 	= [;]	Indicate the position of the parameters that need to be estimated 
									in the parameters vector as a column vector.
				priors		= (;)   List the priors of the parameters that need to be estimated as adapt
									named tuple. (iv = Initial value, lb = lower boud, ub = upper bound, 
									d = distribution)

			# Variables
				@vars 				List all the variables in the model to generate them as symbols.
				x   =	[;]			List in a column vector all the states in the model at time t.
				y   = 	[;]			List in a column vector all the jumpers in the model at time t.
				xp  = 	[;]			List in a column vector all the states in the model at time t+1.
				yp  = 	[;]			List in a column vector all the jumpers in the model at time t+1.

			# Shock
				@vars 				List all the exogenous shocks affecting the model at time t to 
									generate them as symbols.
				e   =   [;]			List in a column vector all the exogenous shocks affecting the 
									model at time t.
				eta =   Array([])

			# Equilibrium conditions
				f1  =   			Write each of the equilibrium conditions as symbolic functions.
				f2  =               
				... = 	

				f   =   [f1;f2;f3]	List in a column vector all the equilibrium conditions. 

			# Steady state (OPTIONAL)
				SS = [;] 		If known, provide the symbolic expresion of the steady state in 
								terms of the parameters.
								It should be provided as a column vector, in which the variables 
								should be ordered in the following order: [x; y; xp; yp].
								If the steady state is unknown leave it as an empty vector (SS=[]), 
								so that the program tries to estimate it.
		
		## Solution
			# Parametrization
				PAR     =   [;]		List in a column vector the specific value of all the parameters 
									in the model. 
									The order should coincide with that of the parameters. vector.
									An initial value for the estimated parameters must be also included 
									in the vector.
		
		## Estimation
			c       =
			npart   =          		State the number of particles in the SMC algorithm.
			nphi    =     			State the number of stages in the SMC algorithm.
			lam     =              	State the value of the bending coefficient.