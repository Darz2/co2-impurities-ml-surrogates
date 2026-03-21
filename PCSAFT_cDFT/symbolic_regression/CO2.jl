# export JULIA_NUM_THREADS=2

println(Sys.CPU_THREADS)
println("Number of threads = ", Threads.nthreads())

using CSV, DataFrames, Random, Plots, Statistics
using SymbolicRegression
using Serialization
# using Pluto

cd(@__DIR__)
df = CSV.read("pure_component_results.csv", DataFrame)
df[!, :Tr] = df[!, :T_K] ./ df[!, :Tc]
df[!, :Pr] = df[!, :Psat] ./ df[!, :Pc]

# =========================
# Helper functions
# =========================

function rmse(y, yhat)
    return sqrt(mean((y .- yhat).^2))
end

function r2_score(y, yhat)
    ss_res = sum((y .- yhat).^2)
    ss_tot = sum((y .- mean(y)).^2)
    return 1 - ss_res / ss_tot
end


target      = ["gamma"]
exclude     = vcat(["rhoL", "rhoV", "Tc", "Pc", "Psat"],target)
features    = [col for col in names(df) if col ∉ exclude]
println("Features: ", features)

X           = Matrix{Float64}(df[:, features])
y_gamma     = Vector{Float64}(df[:, "gamma"])

# =========================
# Train / test split (80/20)
# =========================
n           = size(X, 1)
idx         = randperm(MersenneTwister(5002034), n)
n_train     = round(Int, 0.8 * n)
train_idx   = idx[1:n_train]
test_idx    = idx[n_train+1:end]
X_train     = X[train_idx, :]
X_test      = X[test_idx, :]

y_train_gamma = y_gamma[train_idx]
y_test_gamma  = y_gamma[test_idx]

println("Training: $(size(X_train, 1)) rows")
println("Testing:  $(size(X_test, 1)) rows")

# =========================
# Custom operators
# =========================
oneminusx(x::T) where {T} = one(T) - x

# =========================
# Symbolic regression options
# =========================
options = Options(;
    binary_operators = (+, -, *, /),
    unary_operators  = (oneminusx),
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

# =========================
# Choose equation from Pareto frontier
# =========================
dominating_gamma = calculate_pareto_frontier(hall_of_fame_gamma)
best_idx      = min(3, length(dominating_gamma))
# best_idx         = 10
best_eq_gamma    = dominating_gamma[best_idx]

println("\nChosen/best equation for gamma:")
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
    xlabel = "Actual γ",
    ylabel = "Predicted γ",
    label = "Test data",
    markersize = 4,
    title = "Parity Plot for γ",
    legend = :topleft)

plot!(
    p_gamma,
    [xmin_gamma, xmax_gamma],
    [xmin_gamma, xmax_gamma],
    label = "y = x",
    lw = 2,
    color = :black,
    line = :dash)

savefig(p_gamma, "outputs/parity_plot_gamma.png")