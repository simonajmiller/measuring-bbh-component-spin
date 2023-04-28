## functions used in the inspect_results notebook 


def find_all_missing(pop, events):
    
    missing = []
    not_missing = []
    
    for job in events: 
        JOB=int(job) 

        f = '../{1}/job_{0:05d}_result.json'.format(JOB, pop)
        file_exists = exists(f)
        if not file_exists:
            missing += [JOB]
            #print(JOB)
        else:
            try:
                # Read file
                with open(f,'r') as jf:
                    result = json.load(jf)
                chiEff = np.array(result['posterior']['content']['chi_eff'])
                not_missing += [JOB]

            except Exception as e:
                print(JOB)
                missing += [JOB]
                print(e)
            
    return missing, not_missing


def reflective_kde_on_grid(points, grid):
    
    A = grid[0]
    B = grid[-1]
    
    kde_on_grid = gaussian_kde(points)(grid) + gaussian_kde(-points + 2*A)(grid) + gaussian_kde(-points + 2*B)(grid)
    
    return kde_on_grid