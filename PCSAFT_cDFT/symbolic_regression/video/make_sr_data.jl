using MLJ
using SymbolicRegression
using JSON3
using Random

cd(@__DIR__)

# -------------------------------
# 1. Create synthetic data
# -------------------------------
Random.seed!(1234)

n = 80
x = collect(range(-2.0, 2.0, length=n))

# True relation:
# y = 2x^2 + 1 + small noise
y = 2 .* x.^2 .+ 1 .+ 0.1 .* randn(n)

# MLJ table input:
X = (x = x,)

# -------------------------------
# 2. Set up symbolic regression
# -------------------------------
SRRegressor = @load SRRegressor pkg=SymbolicRegression verbosity=0

model = SRRegressor(
    niterations=80,
    binary_operators=[+, -, *],
    unary_operators=[],
)

mach = machine(model, X, y)
fit!(mach)

# -------------------------------
# 3. Inspect results
# -------------------------------
r = report(mach)

println("Best equation index: ", r.best_idx)
println("Best equation: ", r.equation_strings[r.best_idx])

# -------------------------------
# 4. Build export data
# -------------------------------
# r.equation_strings contains discovered equations.
# predict(mach, (data=X, idx=i)) evaluates equation i.

items = Any[]

for i in eachindex(r.equation_strings)
    yhat = predict(mach, (data=X, idx=i))

    push!(items, Dict(
        "idx" => i,
        "equation" => r.equation_strings[i],
        "is_best" => (i == r.best_idx),
        "x_plot" => collect(x),
        "y_plot" => collect(yhat),
    ))
end

output = Dict(
    "title" => "Symbolic Regression Demo",
    "x_data" => collect(x),
    "y_data" => collect(y),
    "best_idx" => r.best_idx,
    "best_equation" => r.equation_strings[r.best_idx],
    "equations" => items,
)

open("sr_video_data.json", "w") do io
    JSON3.pretty(io, output)
end

println("Saved sr_video_data.json")