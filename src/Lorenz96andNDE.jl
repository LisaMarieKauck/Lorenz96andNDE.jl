module Lorenz96andNDE

using DifferentialEquations
function lorenz96!(dy,y,p,t) 
    F, K = p
    
    dy[1]=(y[2]-y[K-1])*y[K]-y[1]+F;
    dy[2]=(y[3]-y[K])*y[1]-y[2]+F;
    dy[K]=(y[1]-y[K-2])*y[K-1]-y[K]+F;
    for j=3:K-1
        dy[j]=(y[j+1]-y[j-2])*y[j-1]-y[j]+F;
    end
end
K = 36 # so 10 degrees of longtitude per node
F = 0.5

u0 = rand(K);
prob = ODEProblem(lorenz96!, u0, (0.,100.), (F, K));
sol = solve(prob);

end
