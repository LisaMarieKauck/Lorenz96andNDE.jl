module Lorenz96andNDE

using DifferentialEquations, Flux, DiffEqFlux, Plots, NODEData
using  CUDA, OrdinaryDiffEq, BenchmarkTools
using  Random, JLD2
using ChaoticNDETools

function lorenz96!(dy,y,p,t) 
    F, K = p
    
    dy[1]=(y[2]-y[K-1])*y[K]-y[1]+F;
    dy[2]=(y[3]-y[K])*y[1]-y[2]+F;
    dy[K]=(y[1]-y[K-2])*y[K-1]-y[K]+F;
    for j=3:K-1
        dy[j]=(y[j+1]-y[j-2])*y[j-1]-y[j]+F;
    end
end
K = 36; # so 10 degrees of longtitude per node
F = 0.5;
dt=0.1;
N_t=500;
N_t_train = N_t
N_t_valid = N_t_train*3
N_t = N_t_train + N_t_valid
t_transient=100.;
tspan=(0., t_transient+N_t*0.1)
u0 = rand(K);

prob = ODEProblem(lorenz96!, u0, tspan, (F, K));
sol = solve(prob, Tsit5(), saveat=t_transient:dt:t_transient + N_t * dt)

t_train = t_transient:dt:t_transient+N_t_train*dt
data_train = DeviceArray(sol(t_train))

t_valid = t_transient+N_t_train*dt:dt:t_transient+N_t_train*dt+N_t_valid*dt
data_valid = DeviceArray(sol(t_valid))

#train, valid = NODEDataloader(sol, 10; dt=dt, valid_set=0.8)
train = NODEDataloader(Float32.(data_train), t_train, 2)
valid = NODEDataloader(Float32.(data_valid), t_valid, 2)

N_WEIGHTS = 15
nn = Chain(Dense(2, N_WEIGHTS, swish), Dense(N_WEIGHTS, N_WEIGHTS, swish), Dense(N_WEIGHTS, N_WEIGHTS, swish), Dense(N_WEIGHTS, 2)) |> gpu
p, re_nn = Flux.destructure(nn)

neural_ode(u, p, t) = re_nn(p)(u)
node_prob = ODEProblem(neural_ode, u0, (Float32(0.),Float32(dt)), p)

predict(t, u0; reltol=1e-5) = DeviceArray(solve(remake(node_prob; tspan=(t[1],t[end]),u0=u0, p=p), Tsit5(), dt=dt, saveat = t, reltol=reltol))

loss(t, u0) = sum(abs2, predict(t, view(u0,:,1)) - u0)

function plot_node()
    plt = plot(valid.t, Array(predict(valid.t,valid.data[:,1])'), label="Neural ODE")
    plot!(plt, valid.t, valid.data', label="Training Data",xlims=[125,150])
    display(plt)
end
plot_node()

opt = Flux.AdamW(1f-3)

Flux.train!(loss, Flux.params(p), train, opt)
plot_node()

λ_max = 0.9056 # maximum LE of the L63

TRAIN = true
if TRAIN 
    println("starting training...")
    for i_e = 1:100
        Flux.train!(loss, Flux.params(p), train, opt)
        #plot_node()
        δ = ChaoticNDETools.forecast_δ(Array(predict(valid.t,valid.data[:,1])), valid.data)
        forecast_length = findall(δ .> 0.4)[1][2] * dt * λ_max
        println("forecast_length=", forecast_length)

        if (i_e % 30) == 0  # reduce the learning rate every 30 epochs
            opt[1].eta /= 2
        end
    end
end
