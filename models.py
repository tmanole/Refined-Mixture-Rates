n_num = 100                          
ns = np.concatenate([np.linspace(100, 900, 5), np.linspace(1000, 100000, n_num-5)]) 

if model == 1:    
    d =2
    K0=2

    mix_type = "si"
        
    def get_params(n):
        theta0 = np.array([[0,0],[0.2,0.2]])
        sigma0 = np.array([0.01*np.identity(d) for _ in range(K0)])
        pi0    = np.array([0.5, 0.5])  

        return (theta0, sigma0, pi0)


elif model == 2:
    d =2
    K0=3

    mix_type = "weak"
        
    def get_params(n):
        theta0 = np.array([[0,3],[1,-4],[5,2]]) / 10.0
        sigma0 = np.array([np.array([[4.2824, 1.7324],[1.7324,0.81759]]), np.array([[1.75,-1.25],[-1.25,1.75]]), np.array([[1,0],[0,4]]) ]) / 100.0 
        pi0    = np.array([0.3, 0.4, 0.3])  

        return (theta0, sigma0, pi0)

elif model == 3:    
    d =1
    K0=3
    mix_type = "si"
        
    def get_params(n):
        epsilon_n = n**(-1.0/(4*K0-6))
        theta0 = np.array([[0],[0.2+epsilon_n], [0.2-1.5*epsilon_n]])
        sigma0 = np.array([0.01 * np.identity(d) for _ in range(K0)])
        pi0    = np.array([1.0/3, 1.0/3, 1.0/3])  

        return (theta0, sigma0, pi0)

elif model == 4:    
    d =1
    K0=4
    mix_type = "si"
        
    def get_params(n):
        epsilon_n = n**(-1.0/(4*K0-6))
        theta0 = np.array([[0],[0.2+epsilon_n], [0.2+4*epsilon_n], [0.2-1.5*epsilon_n]])
        sigma0 = np.array([0.01 * np.identity(d) for _ in range(K0)])
        pi0    = np.array([0.25, 0.25, 0.25, 0.25])  

        return (theta0, sigma0, pi0)

else:
    sys.exit("Model unrecognized.")

theta_star = np.array([[0],[.2]])
pi_star    = np.array([0.5, 0.5])  

def get_xi(n):
    return np.log(n)
