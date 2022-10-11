module Lorenz96andNDE

import Pkg
Pkg.activate(".")
using DynamicalSystems, CairoMakie 
using DifferentialEquations, Flux, DiffEqFlux, NODEData
using  CUDA, OrdinaryDiffEq, BenchmarkTools
using  Random, JLD2
using ChaoticNDETools
using Plots

# parameters and initial conditions
K = 5
J = 8
F = 8
p=[F, 0.1, 0.2, 0.1]
x0 = F* ones(K)
y0= F* ones(K*J)
u0=[x0,y0]
#u0 = F * ones(K+J*K)
u0[1][1] += 0.01 # small perturbation

# Values for trajectory
dt = 0.01 # sampling time
Tf = 30.0 # final time

# predefined Lorenz-96 model from DynamicalSystems.jl
ds = Systems.lorenz96(K; F = F) #todo google how to 2dim
data=trajectory(ds, Tf; dt = dt)
fig = CairoMakie.Figure(resolution = (500, 500))
ax = Axis(fig[1, 1])
lines!(ax, data[:,1], data[:,2], linewidth=1.0)
fig
#save("Lorenz96in2D.png", fig)

# maximum lyapunov exponent
λ_max = lyapunov(ds, 1000, dt=dt)

# Equivalently an explicit version to proceed
function Lorenz96(dX, X, p, t)
    x, y = X
    dx, dy = dX
    F, h, c, b = p
    #slow-grid calculation
    # 3 edge cases explicitly 
     dx[1] = (x[2] - x[K - 1]) * x[K] - x[1] + F - h*c/b * sum(y[1:J])
     dx[2] = (x[3] - x[K]) * x[1] - x[2] + F - h*c/b * sum(y[(J+1):2*J])
     dx[K] = (x[1] - x[K - 2]) * x[K - 1] - x[K] + F - h*c/b * sum(y[(J*(K-1)+1):K*J])
    # general case
    for n = 3:(K - 1)
       dx[n] = (x[n + 1] - x[n - 2]) * x[n - 1] - x[n] + F - h*c/b * sum(y[(J*(n-1)+1):n*J])
    end
    #fast-grid calculation
    # 3 edge cases explicitly 
    dy[1] = -c*b*y[2]*(y[3]-y[J*K])-c*y[1]+F+h*c/b*x[1]
    dy[J*K-1] = -c*b*y[J*K]*(y[1]-y[J*K-2])-c*y[J*K-1]+F+h*c/b*x[(J*K-2)/J+1]
    dy[J*K] = -c*b*y[1]*(y[2]-y[J*K-1])-c*y[J*K]+F+h*c/b*x[(J*K-1)/J+1]
    # general case
    for j = 2:(J*K - 2)
        dy[j] = -c*b*y[j+1]*(y[j+2]-y[j-1])-c*y[j]+F+h*c/b*x[(j-1)/J+1]
    end
end

ds = ContinuousDynamicalSystem(Lorenz96, u0, p)
tr = trajectory(ds, Tf; dt = dt)
fig = CairoMakie.Figure(resolution = (500, 500))
ax = Axis(fig[1, 1])
lines!(ax, tr[:,1], tr[:,2], linewidth=1.0)
fig

#splitting data into training and validation
N_t=500;
N_t_train = N_t
N_t_valid = N_t_train*3
N_t = N_t_train + N_t_valid
t_transient=100.;
N_epochs = 20
tspan=(0., t_transient+N_t*0.1)

prob = ODEProblem(Lorenz96, u0, tspan, (F, K));
sol = solve(prob, Tsit5(), saveat=t_transient:dt:t_transient + N_t * dt)

Plots.heatmap(Array(sol))
save("Lorenz96heatmapk5.png", Plots.heatmap(Array(sol)))

t_train = t_transient:dt:t_transient+N_t_train*dt
data_train = DeviceArray(sol(t_train)) #TODO subsetting to X

t_valid = t_transient+N_t_train*dt:dt:t_transient+N_t_train*dt+N_t_valid*dt
data_valid = DeviceArray(sol(t_valid))

train = NODEDataloader(Float32.(data_train), t_train, 2)
valid = NODEDataloader(Float32.(data_valid), t_valid, 2)

#neural network
N_WEIGHTS = 10
nn = Chain(Dense(K, N_WEIGHTS, swish), Dense(N_WEIGHTS, N_WEIGHTS, swish), Dense(N_WEIGHTS, N_WEIGHTS, swish), Dense(N_WEIGHTS, K)) |> gpu
p, re_nn = Flux.destructure(nn)

#substituting small-grid with neural network
function neural_l96(dx,x,p,t)
    F = p[1]
    # 3 edge cases explicitly 
     dx[1] = (x[2] - x[K - 1]) * x[K] - x[1] + F - re_nn(p)(x)[1]
     dx[2] = (x[3] - x[K]) * x[1] - x[2] + F - re_nn(p)(x)[2]
     dx[K] = (x[1] - x[K - 2]) * x[K - 1] - x[K] + F - re_nn(p)(x)[K]
    # general case
    for n = 3:(K - 1)
       dx[n] = (x[n + 1] - x[n - 2]) * x[n - 1] - x[n] + F - re_nn(p)(x)[n]
    end
end
node_prob = ODEProblem(neural_l96, u0, (Float32(0.),Float32(dt)), p) #todo u0 anpassen

predict(t, u0; reltol=1e-5) = DeviceArray(solve(remake(node_prob; tspan=(t[1],t[end]),u0=u0, p=p), Tsit5(), dt=dt, saveat = t, reltol=reltol))

loss(t, u0) = sum(abs2, predict(t, view(u0,:,1)) - u0)

opt = Flux.AdamW(1f-3)

Flux.train!(loss, Flux.params(p), train, opt)

τ_max = 2
N_epochs=10

TRAIN = true
if TRAIN 
    println("starting training...")

    for i_τ = 2:τ_max

        train_τ = NODEDataloader(train, i_τ)
        N_e = N_epochs
    
        for i_e=1:N_e 
            Flux.train!(loss, Flux.params(p), train_τ, opt)

            if (i_e % 5) == 0
                train_error = mean([loss(train[i]...) for i=1:NN_train])
                valid_error = mean([loss(valid[i]...) for i=1:NN_valid])
              
                println("AdamW, i_τ=", i_τ, "- training error =",train_error, "- valid error=", valid_error)

                δ = ChaoticNDETools.forecast_δ(Array(predict(valid.t,valid.data[:,1])), valid.data)
                forecast_length = findall(δ .> 0.4)[1][2] * dt * λ_max
                println("forecast_length=", forecast_length)
            end

            if (i_e % 5) == 0  # reduce the learning rate every 30 epochs
                opt[1].eta /= 2
            end
        end  
       
        
     end
end

# for comparison: neural ODE as rhs

N_WEIGHTS = 10
nn = Chain(Dense(K, N_WEIGHTS, swish), Dense(N_WEIGHTS, N_WEIGHTS, swish), Dense(N_WEIGHTS, N_WEIGHTS, swish), Dense(N_WEIGHTS, K)) |> gpu
p, re_nn = Flux.destructure(nn)

neural_ode(u, p, t) = re_nn(p)(u)
node_prob = ODEProblem(neural_ode, u0, (Float32(0.),Float32(dt)), p)

Flux.train!(loss, Flux.params(p), train, opt)

TRAIN = true
if TRAIN 
    println("starting training...")
    for i_e = 1:100
        Flux.train!(loss, Flux.params(p), train, opt)
        δ = ChaoticNDETools.forecast_δ(Array(predict(valid.t,valid.data[:,1])), valid.data)
        forecast_length = findall(δ .> 0.4)[1][2] * dt * λ_max
        println("forecast_length=", forecast_length)

        if (i_e % 30) == 0  # reduce the learning rate every 30 epochs
            opt[1].eta /= 2
        end
    end
end