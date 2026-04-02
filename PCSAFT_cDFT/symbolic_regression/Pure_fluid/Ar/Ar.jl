# export JULIA_NUM_THREADS=2

println("CPU threads available = ", Sys.CPU_THREADS)
println("Julia threads         = ", Threads.nthreads())

using CSV
using DataFrames
using Random
using Statistics
using Plots
using Serialization
using SymbolicRegression

cd(@__DIR__)
mkpath("outputs")

include("../sr_utils.jl")

# ============================================================
# Data loading and feature engineering
# ============================================================

df = CSV.read("pure_component_results.csv", DataFrame)

df[!, :Tr] = df[!, :T_K] ./ df[!, :Tc]
df[!, :F1] = 1 .- df[!, :Tr]
df[!, :Pr] = df[!, :Psat] ./  df[!, :Pc]

Tc_val = df[1, :Tc]
Pc_val = df[1, :Pc]
println("Tc (PC-SAFT+cDFT) = ", Tc_val, " K")
println("Pc (PC-SAFT+cDFT) = ", Pc_val, " bar")

target = :gamma
exclude = [:rhoL, :rhoV, :Tc, :Pc, :Psat, :T_K, :Tr, target]
features = [col for col in Symbol.(names(df)) if col ∉ exclude]

println("\nFeatures used: ", features)

X = Matrix{Float64}(df[:, features])
y = Vector{Float64}(df[:, target])

# ============================================================
# Train/validation/test split
# ============================================================

n = size(X, 1)
rng = MersenneTwister(5002034)
idx = randperm(rng, n)

n_train = round(Int, 0.70 * n)
n_val   = round(Int, 0.15 * n)

train_idx = idx[1:n_train]
val_idx   = idx[n_train+1:n_train+n_val]
test_idx  = idx[n_train+n_val+1:end]

X_train = X[train_idx, :]
X_val   = X[val_idx, :]
X_test  = X[test_idx, :]

y_train = y[train_idx]
y_val   = y[val_idx]
y_test  = y[test_idx]

println("Training rows   = ", size(X_train, 1))
println("Validation rows = ", size(X_val, 1))
println("Testing rows    = ", size(X_test, 1))

# ============================================================
# Symbolic regression options
# ============================================================

options = Options(
    binary_operators = (+, *, safepow),
    populations      = 50,
    maxsize          = 30,
    parsimony        = 0.005f0,
)

# ============================================================
# Train symbolic regression
# ============================================================

println("\n══ Fitting gamma — Ar ══")

hall_of_fame_gamma = equation_search(
    X_train', y_train;
    options        = options,
    niterations    = 100,
    variable_names = string.(features),
    parallelism    = :multithreading,
)

serialize("outputs/hall_of_fame_gamma.jls", hall_of_fame_gamma)

println("\nSaved hall of fame to:")
println("outputs/hall_of_fame_gamma.jls")

println("\nHall of fame summary:")
println(hall_of_fame_gamma)

# ============================================================
# Choose best equation using validation RMSE
# ============================================================

dominating_gamma = calculate_pareto_frontier(hall_of_fame_gamma)
select_best_equation(dominating_gamma, X_val, y_val, options)

best_idx = 3 # <-- CHANGE THIS INDEX BASED ON THE OUTPUT ABOVE
best_eq_gamma = dominating_gamma[best_idx]
best_eq_str = string_tree(best_eq_gamma.tree, options)

println("\nChosen equation index = ", best_idx)
println("Chosen equation:")
println(best_eq_gamma)

println("\nEquation string:")
println(best_eq_str)

# ============================================================
# Evaluate chosen equation
# ============================================================

yhat_train, ok_train = eval_tree_array(best_eq_gamma.tree, X_train', options)
yhat_val,   ok_val   = eval_tree_array(best_eq_gamma.tree, X_val',   options)
yhat_test,  ok_test  = eval_tree_array(best_eq_gamma.tree, X_test',  options)

println("\nEvaluation successful (train): ", ok_train)
println("Evaluation successful (val):   ", ok_val)
println("Evaluation successful (test):  ", ok_test)

print_metrics("Train", y_train, yhat_train)
print_metrics("Validation", y_val, yhat_val)
print_metrics("Test", y_test, yhat_test)

rmse_train = rmse(y_train, yhat_train)
rmse_val   = rmse(y_val, yhat_val)
rmse_test  = rmse(y_test, yhat_test)

r2_train = r2_score(y_train, yhat_train)
r2_val   = r2_score(y_val, yhat_val)
r2_test  = r2_score(y_test, yhat_test)

# ============================================================
# Save predictions
# ============================================================

CSV.write("outputs/train_predictions_gamma.csv", DataFrame(Actual=y_train, Predicted=yhat_train))
CSV.write("outputs/validation_predictions_gamma.csv", DataFrame(Actual=y_val, Predicted=yhat_val))
CSV.write("outputs/test_predictions_gamma.csv", DataFrame(Actual=y_test, Predicted=yhat_test))

println("\nSaved prediction tables:")
println("outputs/train_predictions_gamma.csv")
println("outputs/validation_predictions_gamma.csv")
println("outputs/test_predictions_gamma.csv")

# ============================================================
# Parity plot
# ============================================================

xmin = min(
    minimum(y_train), minimum(yhat_train),
    minimum(y_val),   minimum(yhat_val),
    minimum(y_test),  minimum(yhat_test)
)

xmax = max(
    maximum(y_train), maximum(yhat_train),
    maximum(y_val),   maximum(yhat_val),
    maximum(y_test),  maximum(yhat_test)
)

p_gamma = scatter(
    y_train, yhat_train;
    xlabel = "Actual γ / [mN/m]",
    ylabel = "Predicted γ / [mN/m]",
    label = "Train",
    markersize = 4,
    title = "Parity Plot for γ — Ar",
    legend = :topleft,
)

scatter!(p_gamma, y_val,  yhat_val;  label = "Validation", markersize = 4)
scatter!(p_gamma, y_test, yhat_test; label = "Test", markersize = 4)

plot!(
    p_gamma,
    [xmin, xmax], [xmin, xmax];
    label = "y = x",
    lw = 2,
    color = :black,
    line = :dash,
)

annotate!(p_gamma, xmin + 0.52*(xmax - xmin), xmax - 0.72*(xmax - xmin),
    text("Chosen Eq. index = $best_idx", 9, :left))
annotate!(p_gamma, xmin + 0.52*(xmax - xmin), xmax - 0.78*(xmax - xmin),
    text("Train: RMSE = $(round(rmse_train, sigdigits=2)), R² = $(round(r2_train, sigdigits=2))", 9, :left))
annotate!(p_gamma, xmin + 0.52*(xmax - xmin), xmax - 0.84*(xmax - xmin),
    text("Val: RMSE = $(round(rmse_val, sigdigits=2)), R² = $(round(r2_val, sigdigits=2))", 9, :left))
annotate!(p_gamma, xmin + 0.52*(xmax - xmin), xmax - 0.90*(xmax - xmin),
    text("Test: RMSE = $(round(rmse_test, sigdigits=2)), R² = $(round(r2_test, sigdigits=2))", 9, :left))

savefig(p_gamma, "outputs/parity_plot_gamma.png")

println("\nSaved parity plot:")
println("outputs/parity_plot_gamma.png")

# ============================================================
# Render equations and export TeX
# ============================================================

render_hall_of_fame(
    "outputs/hall_of_fame_gamma.jls";
    n_best=10,
    digits=4,
    feature_map=Dict("F1" => "(1 - Tr)")
)

export_hall_of_fame_tex(
    "outputs/hall_of_fame_gamma.jls",
    "outputs/equations_gamma.tex";
    n_best=10,
    digits=4,
    feature_map=Dict("F1" => "(1 - Tr)"),
    title="Surface tension correlations from symbolic regression — Ar"
)

println("\nSaved TeX file:")
println("outputs/equations_gamma.tex")
