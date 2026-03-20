# export JULIA_NUM_THREADS=2  # Set this in the terminal before running the script

println(Sys.CPU_THREADS)
println("Number of threads = ", Threads.nthreads())

using CSV, DataFrames, Random, Plots, Statistics
using SymbolicRegression
using Serialization

cd(@__DIR__)

df = CSV.read("CSV_OLD/interfacial_results_cleaned.csv", DataFrame)
df[!, :Tr] = df[!, :temperature] ./ df[!, :Tc]
df[!, :Pr] = df[!, :pressure]    ./ df[!, :Pc]

# =========================
# Target and feature selection
# =========================
target = ["gamma"]

exclude = vcat(
    ["interfacial_thickness", "E_carbon dioxide", "E_argon", "E_hydrogen"],
    ["x_carbon dioxide", "x_argon", "x_hydrogen", "y_carbon dioxide", "y_argon", "y_hydrogen"],
    ["z_carbon dioxide"],
    ["liquid_density", "vapor_density"],  
    ["P_bubble", "P_dew"],
    ["Tc", "Pc"],
    ["temperature", "pressure"],
    target,
)

features = [col for col in names(df) if col ∉ exclude]
println("Features: ", features)

X = Matrix{Float64}(df[:, features])
y_gamma = Vector{Float64}(df[:, "gamma"])

# =========================
# Train / test split (80/20, seed=85)
# =========================
n = size(X, 1)
idx = randperm(MersenneTwister(85), n)
n_train = round(Int, 0.8 * n)

train_idx = idx[1:n_train]
test_idx  = idx[n_train+1:end]

X_train = X[train_idx, :]
X_test  = X[test_idx, :]

y_train_gamma = y_gamma[train_idx]
y_test_gamma  = y_gamma[test_idx]

println("Training: $(size(X_train, 1)) rows")
println("Testing:  $(size(X_test, 1)) rows")

# =========================
# Simple custom operators
# =========================
square(x::T)   where {T} = x * x
cube(x::T)     where {T} = x * x * x
safesqrt(x::T) where {T} = x >= zero(T) ? sqrt(x) : T(NaN)
safelog(x::T)  where {T} = x > zero(T) ? log(x) : T(NaN)
safeexp(x::T)  where {T} = exp(x)
oneminusx(x::T) where {T} = one(T) - x

# =========================
# Simple options for gamma
# =========================
options = Options(;
    binary_operators = (+, -, *, /),
    unary_operators  = (square, cube, safesqrt, safelog, safeexp, oneminusx),
    populations      = 50,
    maxsize          = 25,
    parsimony        = 0.002f0,
)

# =========================
# Run symbolic regression for gamma
# =========================
println("\n══ Fitting gamma ══")

hall_of_fame_gamma = equation_search(
    X_train', y_train_gamma;
    options = options,
    niterations = 100,
    variable_names = features,
    parallelism = :multithreading,
)

println("\nBest equations for gamma:")
println(hall_of_fame_gamma)

serialize("outputs/hall_of_fame_gamma.jls", hall_of_fame_gamma)
println("\nSaved: outputs/hall_of_fame_gamma.jls")

# =========================
# Helper metrics
# =========================
rmse(y, yhat) = sqrt(mean((y .- yhat).^2))

function r2_score(y, yhat)
    ss_res = sum((y .- yhat).^2)
    ss_tot = sum((y .- mean(y)).^2)
    return 1 - ss_res / ss_tot
end

# =========================
# Choose equation from Pareto frontier
# =========================
dominating_gamma = calculate_pareto_frontier(hall_of_fame_gamma)
# best_idx = min(3, length(dominating_gamma))
best_idx = 10
best_eq_gamma = dominating_gamma[best_idx]

println("\nChosen equation for gamma:")
println(best_eq_gamma)
println("\nEquation string for gamma:")
println(string_tree(best_eq_gamma.tree, options))

# =========================
# Evaluate equation
# =========================
yhat_train_gamma, ok_train_gamma = eval_tree_array(best_eq_gamma.tree, X_train', options)
yhat_test_gamma,  ok_test_gamma  = eval_tree_array(best_eq_gamma.tree, X_test',  options)

println("\nGamma evaluation successful (train): ", ok_train_gamma)
println("Gamma evaluation successful (test):  ", ok_test_gamma)

# =========================
# Metrics
# =========================
rmse_train_gamma = rmse(y_train_gamma, yhat_train_gamma)
rmse_test_gamma  = rmse(y_test_gamma,  yhat_test_gamma)

r2_train_gamma = r2_score(y_train_gamma, yhat_train_gamma)
r2_test_gamma  = r2_score(y_test_gamma,  yhat_test_gamma)

println("\n══ Metrics for gamma ══")
println("Train RMSE = ", rmse_train_gamma)
println("Test  RMSE = ", rmse_test_gamma)
println("Train R²   = ", r2_train_gamma)
println("Test  R²   = ", r2_test_gamma)

# =========================
# Parity plot
# =========================
xmin_gamma = min(minimum(y_test_gamma), minimum(yhat_test_gamma))
xmax_gamma = max(maximum(y_test_gamma), maximum(yhat_test_gamma))

p_gamma = scatter(
    y_test_gamma, yhat_test_gamma,
    xlabel = "Actual gamma",
    ylabel = "Predicted gamma",
    label = "Test data",
    markersize = 4,
    title = "Parity Plot for gamma",
    legend = :topleft
)

plot!(
    p_gamma,
    [xmin_gamma, xmax_gamma],
    [xmin_gamma, xmax_gamma],
    label = "y = x",
    lw = 2
)

savefig(p_gamma, "outputs/parity_plot_gamma.png")