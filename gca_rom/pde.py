import numpy as np
import ipdb


def problem(argument):
    """
    problem(argument: int) -> Tuple[str, str, np.ndarray, np.ndarray]

    This function takes in an integer argument and returns a tuple containing the problem name (str), variable name (str), mu1 (np.ndarray) and mu2 (np.ndarray) for the specified case.

    The possible values of argument are:

    Poisson problem
    Advection problem
    Graetz problem
    Navier-Stokes problem (variable = VX)
    Navier-Stokes problem (variable = VY)
    Navier-Stokes problem (variable = P)
    Diffusion problem
    Poiseuille problem
    Linear elasticity problem
    Returns:
    Tuple containing the problem name (str), variable name (str), mu1 (np.ndarray) and mu2 (np.ndarray) for the specified case.
    """

    match argument:
        case 1:
            problem_name = "poisson"
            variable = 'U'
            mu1 = np.linspace(0.01, 10., 10)
            mu2 = np.linspace(0.01, 10., 10)
            mu_space = [mu1, mu2]
            n_param = 2
        case 2:
            problem_name = "advection"
            variable = 'U'
            mu1 = np.linspace(0., 6., 10)
            mu2 = np.linspace(-1.0, 1.0, 10)
            mu_space = [mu1, mu2]
            n_param = 2
        case 3:
            problem_name = "graetz"
            variable = 'U'
            mu1 = np.linspace(1., 3., 10)
            mu2 = np.linspace(0.01, 0.1, 20)
            mu_space = [mu1, mu2]
            n_param = 2
        case 4:
            problem_name = "navier_stokes"
            variable = 'VX'
            mu1 = np.linspace(0.5, 2., 21)[::2]
            mu2 = np.linspace(2., 0.5, 151)[::5]
            mu_space = [mu1, mu2]
            n_param = 2
        case 5:
            problem_name = "navier_stokes"
            variable = 'VY'
            mu1 = np.linspace(0.5, 2., 21)[::2]
            mu2 = np.linspace(2., 0.5, 151)[::5]
            mu_space = [mu1, mu2]
            n_param = 2
        case 6:
            problem_name = "navier_stokes"
            variable = 'P'
            mu1 = np.linspace(0.5, 2., 21)[::2]
            mu2 = np.linspace(2., 0.5, 151)[::5]
            mu_space = [mu1, mu2]
            n_param = 2
        case 7:
            problem_name = "diffusion"
            variable = 'U'
            mu1 = np.linspace(0.2, 4., 20)
            mu2 = np.linspace(0., 1., 20)
            mu_space = [mu1, mu2]
            n_param = 2
        case 8:
            problem_name = "poiseuille"
            variable = 'U'
            mu1 = np.linspace(0.5, 10., 20)
            mu2 = np.linspace(0., 1., 50)
            mu_space = [mu1, mu2]
            n_param = 2
        case 9:
            problem_name = "elasticity"
            variable = 'U'
            mu1 = np.linspace(2., 20., 11)
            mu2 = np.linspace(2., 200., 11)
            mu_space = [mu1, mu2]
            n_param = 2
        case 10:
            problem_name = "stokes_u"
            variable = 'U'
            mu_range = [(0.5, 1.5), (0.5, 1.5), (0.5, 1.5), (0.5, 1.5), (0.5, 1.5), (-np.pi/6, np.pi/6), (-10, 10)]
            mu_space = []
            n_pts = [2]*(len(mu_range)-1)+[11]
            for i in range(len(mu_range)):
                mu_space.append(np.linspace(mu_range[i][0], mu_range[i][1], n_pts[i]))
            n_param = 7
        case 11:
            problem_name = "lid_cavity"
            variable = 'U'
            
            """ Fourier expansion approach (4 frequencies)  """
            
            # Load the matrix from alphaM.npy
            # alphaM = np.load('../lid_driven_cavity_fenics/alphaM.npy')
            # K = alphaM.shape[0]
            # n_coeff = alphaM.shape[1]
            # mu_space = []
            # for i in range(n_coeff):
            #     mu_space.append(alphaM[:, i].tolist())
            # mu_time = np.linspace(0.0, 1.8, 10)
            # mu_space.append(mu_time.tolist())
            # n_param = n_coeff + 1
            #ipdb.set_trace()
            
#             alphaM_cols = [alphaM[:, i].tolist() for i in range(alphaM.shape[1])]
            
#             mu1 = alphaM_cols[0]
#             mu2 = alphaM_cols[1]
#             mu3 = alphaM_cols[2]
#             mu4 = alphaM_cols[3]
#             mu5 = np.linspace(0.0,1.8,10)
#             mu_space = [mu1, mu2, mu3, mu4,mu5]
#             n_param = 5
            
            """ Non causal (?) approach  """
            
            u_t_matrix = np.load('../lid_driven_cavity_fenics/u_t_matrix.npy')
            u_t_matrix_rows = [u_t_matrix[i, :].tolist() for i in range(u_t_matrix.shape[0])]
            mu_space = []
            for i in range(u_t_matrix.shape[0]):
                mu_space.append(u_t_matrix_rows[i])
            mu_space.append(np.linspace(0.0, 1.8, 10))
            #ipdb.set_trace()      
            n_param = 2
            
            
            

    return problem_name, variable, mu_space, n_param
