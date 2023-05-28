from Neural_Network_Problem import *

create_dir('results')

#EXPERIMENT 1 - 
evolve_params = {
    'size': 40,
    'file_name': 'ex1_inv_blen',
    'gens': 100,
    'select': tournament_sel,
    'mutate': inversion_mutation,
    'crossover': blend_crossover,
    'xo_p': 0.9,
    'mut_p': 0.2,
    'elitism': True,
    'n_elitism': 1
}

def run_evolution(iteration):
    return networkEval(iteration, evolve_params)

parallel(delayed(run_evolution)(i) for i in n_runs)

#EXPERIMENT 2 - 
evolve_params2 = {
    'size': 40,
    'file_name': 'exp2_inv_arit',
    'gens': 100,
    'select': tournament_sel,
    'mutate': inversion_mutation,
    'crossover': arithmetic_xo,
    'xo_p': 0.9,
    'mut_p': 0.2,
    'elitism': True,
    'n_elitism': 1
}

def run_evolution(iteration):
    return networkEval(iteration, evolve_params2)

parallel(delayed(run_evolution)(i) for i in n_runs)

#EXPERIMENT 3 - 
evolve_params3 = {
    'size': 40,
    'file_name': 'exp3_inv_geo',
    'gens': 100,
    'select': tournament_sel,
    'mutate': inversion_mutation,
    'crossover': geometric_crossover,
    'xo_p': 0.9,
    'mut_p': 0.2,
    'elitism': True,
    'n_elitism': 1
}

def run_evolution(iteration):
    return networkEval(iteration, evolve_params3)

parallel(delayed(run_evolution)(i) for i in n_runs)

#EXPERIMENT 4 - 
evolve_params4 = {
    'size': 40,
    'file_name': 'exp4_scrab_blen',
    'gens': 100,
    'select': tournament_sel,
    'mutate': scramble_mutation,
    'crossover': blend_crossover,
    'xo_p': 0.9,
    'mut_p': 0.2,
    'elitism': True,
    'n_elitism': 1
}

def run_evolution(iteration):
    return networkEval(iteration, evolve_params4)

parallel(delayed(run_evolution)(i) for i in n_runs)

#EXPERIMENT 5 - 
evolve_params5 = {
    'size': 40,
    'file_name': 'exp5_scrab_arit',
    'gens': 100,
    'select': tournament_sel,
    'mutate': scramble_mutation,
    'crossover': arithmetic_xo,
    'xo_p': 0.9,
    'mut_p': 0.2,
    'elitism': True,
    'n_elitism': 1
}

def run_evolution(iteration):
    return networkEval(iteration, evolve_params5)

parallel(delayed(run_evolution)(i) for i in n_runs)

#EXPERIMENT 6 - 
evolve_params6 = {
    'size': 40,
    'file_name': 'exp6_scrab_geo',
    'gens': 100,
    'select': tournament_sel,
    'mutate': scramble_mutation,
    'crossover': geometric_crossover,
    'xo_p': 0.9,
    'mut_p': 0.2,
    'elitism': True,
    'n_elitism': 1
}

def run_evolution(iteration):
    return networkEval(iteration, evolve_params6)

parallel(delayed(run_evolution)(i) for i in n_runs)

#EXPERIMENT 7 - 
evolve_params7 = {
    'size': 40,
    'file_name': 'exp7_geo_blen',
    'gens': 100,
    'select': tournament_sel,
    'mutate': geometric_semantic_mutation,
    'crossover': blend_crossover,
    'xo_p': 0.9,
    'mut_p': 0.2,
    'elitism': True,
    'n_elitism': 1
}

def run_evolution(iteration):
    return networkEval(iteration, evolve_params7)

parallel(delayed(run_evolution)(i) for i in n_runs)


#EXPERIMENT 8 - 
evolve_params8 = {
    'size': 40,
    'file_name': 'exp8_geo_arit',
    'gens': 100,
    'select': tournament_sel,
    'mutate': geometric_semantic_mutation,
    'crossover': arithmetic_xo,
    'xo_p': 0.9,
    'mut_p': 0.2,
    'elitism': True,
    'n_elitism': 1
}

def run_evolution(iteration):
    return networkEval(iteration, evolve_params8)

parallel(delayed(run_evolution)(i) for i in n_runs)

#EXPERIMENT 9 - 
evolve_params9 = {
    'size': 40,
    'file_name': 'exp9_geo_geo',
    'gens': 100,
    'select': tournament_sel,
    'mutate': geometric_semantic_mutation,
    'crossover': geometric_crossover,
    'xo_p': 0.9,
    'mut_p': 0.2,
    'elitism': True,
    'n_elitism': 1
}

def run_evolution(iteration):
    return networkEval(iteration, evolve_params9)

parallel(delayed(run_evolution)(i) for i in n_runs)

#EXPERIMENT 10
evolve_params10 = {
    'size': 40,
    'file_name': 'exp10_inv_arit_elist_FALSE',
    'gens': 100,
    'select': tournament_sel,
    'mutate': inversion_mutation,
    'crossover': arithmetic_xo,
    'xo_p': 0.9,
    'mut_p': 0.2,
    'elitism': False,
    'n_elitism':0
}

def run_evolution(iteration):
    return networkEval(iteration, evolve_params10)

parallel(delayed(run_evolution)(i) for i in n_runs)


#EXPERIMENT 11
evolve_params11 = {
    'size': 40,
    'file_name': 'exp11_inv_arit_elist_5',
    'gens': 100,
    'select': tournament_sel,
    'mutate': inversion_mutation,
    'crossover': arithmetic_xo,
    'xo_p': 0.9,
    'mut_p': 0.2,
    'elitism': True,
    'n_elitism':5
}

def run_evolution(iteration):
    return networkEval(iteration, evolve_params11)

parallel(delayed(run_evolution)(i) for i in n_runs)


#EXPERIMENT 12 - 
evolve_params12 = {
    'size': 40,
    'file_name': 'exp12_scrab_geo_elist_FALSE_ACC',
    'gens': 100,
    'select': tournament_sel,
    'mutate': scramble_mutation,
    'crossover': geometric_crossover,
    'xo_p': 0.9,
    'mut_p': 0.2,
    'elitism': False,
    'n_elitism': 0
}

def run_evolution(iteration):
    return networkEval(iteration, evolve_params12)

parallel(delayed(run_evolution)(i) for i in n_runs)

#EXPERIMENT 13 - 
evolve_params13 = {
    'size': 40,
    'file_name': 'exp13_scrab_geo_elist_5',
    'gens': 100,
    'select': tournament_sel,
    'mutate': scramble_mutation,
    'crossover': geometric_crossover,
    'xo_p': 0.9,
    'mut_p': 0.2,
    'elitism': True,
    'n_elitism': 5
}

def run_evolution(iteration):
    return networkEval(iteration, evolve_params13)

parallel(delayed(run_evolution)(i) for i in n_runs)


#EXPERIMENT 14 - USE BRIER SCORE 
evolve_params14 = {
    'size': 40,
    'file_name': 'exp14_brier',
    'gens': 100,
    'select': tournament_sel,
    'mutate': scramble_mutation,
    'crossover': geometric_crossover,
    'xo_p': 0.9,
    'mut_p': 0.2,
    'elitism': False,
    'n_elitism': 0
}
def run_evolution(iteration):
    return networkEval(iteration, evolve_params14)

parallel(delayed(run_evolution)(i) for i in n_runs)


#EXPERIMENT 15 - 
evolve_params15 = {
    'size': 40,
    'file_name': 'exp15_best_ind',
    'gens': 250,
    'select': tournament_sel,
    'mutate': scramble_mutation,
    'crossover': geometric_crossover,
    'xo_p': 0.9,
    'mut_p': 0.2,
    'elitism': False,
    'n_elitism': 0
}
def run_evolution(iteration):
    return networkEval(iteration, evolve_params15)

parallel(delayed(run_evolution)(i) for i in n_runs)