module Lorenz96andNDE

using DifferentialEquations, NODEData, Flux, DiffEqFlux 
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
dt=0.1
u0 = rand(K);
prob = ODEProblem(lorenz96!, u0, (0.,100.), (F, K));
sol = solve(prob, Tsit5(), saveat=dt);

train, valid = NODEDataloader(sol, 10; dt=dt, valid_set=0.8)

N_WEIGHTS = 10 
nn = Chain(Dense(2, N_WEIGHTS, swish), Dense(N_WEIGHTS, N_WEIGHTS, swish), Dense(N_WEIGHTS, N_WEIGHTS, swish), Dense(N_WEIGHTS, 2)) |> gpu
p, re_nn = Flux.destructure(nn)

neural_ode(u, p, t) = re_nn(p)(u)
node_prob = ODEProblem(neural_ode, x0, (Float32(0.),Float32(dt)), p)

predict(t, u0) = Array(solve(remake(node_prob; tspan=(t[1],t[end]),u0=u0, p=p), Tsit5(), dt=dt, saveat = t))

loss(t, u0) = sum(abs2, predict(t, view(u0,:,1)) - u0)

opt = Flux.AdamW(1f-3)

TRAIN = false
if TRAIN 
    println("starting training...")
    for i_e = 1:400
        Flux.train!(loss, Flux.params(p), train, opt)
        if (i_e % 30) == 0  # reduce the learning rate every 30 epochs
            opt[1].eta /= 2
        end
    end
end

end
