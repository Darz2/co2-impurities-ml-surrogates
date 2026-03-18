using CSV, DataFrames, Random, Plots, Statistics
using SymbolicRegression
using Serialization

cd(@__DIR__)

# =========================
# Custom operators
# Define BEFORE deserialize(...)
# =========================
square(x::T) where {T} = x * x
cube(x::T) where {T} = x * x * x
inv_(x::T) where {T} = one(T) / x
invsquare(x::T) where {T} = one(T) / (x * x)
safesqrt(x::T) where {T} = x >= zero(T) ? sqrt(x) : T(NaN)
safelog(x::T) where {T} = x > zero(T) ? log(x) : T(NaN)

# =========================
# Load saved hall-of-fame files
# =========================
hall_of_fame_bubble = deserialize("outputs/hall_of_fame_bubble.jls")
hall_of_fame_dew    = deserialize("outputs/hall_of_fame_dew.jls")

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
# Load dataset
# =========================
df = CSV.read("CSV_OLD/interfacial_results_cleaned.csv", DataFrame)

target  = ["P_bubble", "P_dew"]
exclude = vcat(
    ["gamma", "interfacial_thickness", "E_carbon dioxide", "E_argon", "E_hydrogen"],
    ["x_carbon dioxide", "x_argon", "x_hydrogen", "y_carbon dioxide", "y_argon", "y_hydrogen"],
    ["z_carbon dioxide"],
    ["liquid_density", "vapor_density"],
    ["pressure"],
    target,
)

features = [col for col in names(df) if col ∉ exclude]
println("Features: ", features)

X = Matrix{Float64}(df[:, features])
y_bubble = Vector{Float64}(df[:, "P_bubble"])
y_dew    = Vector{Float64}(df[:, "P_dew"])

# =========================
# Recreate same train/test split
# =========================
n = size(X, 1)
idx = randperm(MersenneTwister(85), n)
n_train = round(Int, 0.8 * n)
train_idx = idx[1:n_train]
test_idx  = idx[n_train+1:end]

X_train = X[train_idx, :]
X_test  = X[test_idx, :]
y_train_bubble = y_bubble[train_idx]
y_test_bubble  = y_bubble[test_idx]
y_train_dew    = y_dew[train_idx]
y_test_dew     = y_dew[test_idx]

# =========================
# Rebuild options
# =========================
options = Options(;
    binary_operators = (+, -, *, /),
    unary_operators  = (square, cube, inv_, invsquare, safesqrt, safelog, abs),
)

# =========================
# Pareto frontiers
# =========================
dominating_bubble = calculate_pareto_frontier(hall_of_fame_bubble)
dominating_dew    = calculate_pareto_frontier(hall_of_fame_dew)

best_eq_bubble = dominating_bubble[end]
best_eq_dew    = dominating_dew[end]

println("\nChosen equation for P_bubble:")
println(best_eq_bubble)
println("\nEquation string for P_bubble:")
println(string_tree(best_eq_bubble.tree, options))

println("\nChosen equation for P_dew:")
println(best_eq_dew)
println("\nEquation string for P_dew:")
println(string_tree(best_eq_dew.tree, options))

# =========================
# Evaluate equations
# =========================
yhat_train_bubble, ok_train_bubble = eval_tree_array(best_eq_bubble.tree, X_train', options)
yhat_test_bubble,  ok_test_bubble  = eval_tree_array(best_eq_bubble.tree, X_test',  options)

yhat_train_dew, ok_train_dew = eval_tree_array(best_eq_dew.tree, X_train', options)
yhat_test_dew,  ok_test_dew  = eval_tree_array(best_eq_dew.tree, X_test',  options)

println("\nBubble evaluation successful (train): ", ok_train_bubble)
println("Bubble evaluation successful (test):  ", ok_test_bubble)
println("Dew evaluation successful (train):    ", ok_train_dew)
println("Dew evaluation successful (test):     ", ok_test_dew)

# =========================
# Metrics
# =========================
rmse_train_bubble = rmse(y_train_bubble, yhat_train_bubble)
rmse_test_bubble  = rmse(y_test_bubble,  yhat_test_bubble)
r2_train_bubble   = r2_score(y_train_bubble, yhat_train_bubble)
r2_test_bubble    = r2_score(y_test_bubble,  yhat_test_bubble)

rmse_train_dew = rmse(y_train_dew, yhat_train_dew)
rmse_test_dew  = rmse(y_test_dew,  yhat_test_dew)
r2_train_dew   = r2_score(y_train_dew, yhat_train_dew)
r2_test_dew    = r2_score(y_test_dew,  yhat_test_dew)

println("\n══ Metrics for P_bubble ══")
println("Train RMSE = ", rmse_train_bubble)
println("Test  RMSE = ", rmse_test_bubble)
println("Train R²   = ", r2_train_bubble)
println("Test  R²   = ", r2_test_bubble)

println("\n══ Metrics for P_dew ══")
println("Train RMSE = ", rmse_train_dew)
println("Test  RMSE = ", rmse_test_dew)
println("Train R²   = ", r2_train_dew)
println("Test  R²   = ", r2_test_dew)

# =========================
# Parity plot: P_bubble
# =========================
xmin_bubble = min(minimum(y_test_bubble), minimum(yhat_test_bubble))
xmax_bubble = max(maximum(y_test_bubble), maximum(yhat_test_bubble))

p_bubble = scatter(
    y_test_bubble, yhat_test_bubble,
    xlabel = "Actual P_bubble",
    ylabel = "Predicted P_bubble",
    label = "Test data",
    markersize = 4,
    title = "Parity Plot for P_bubble",
    legend = :topleft
)

plot!(
    p_bubble,
    [xmin_bubble, xmax_bubble],
    [xmin_bubble, xmax_bubble],
    label = "y = x",
    lw = 2
)

savefig(p_bubble, "outputs/parity_plot_P_bubble.png")
println("\nSaved: outputs/parity_plot_P_bubble.png")

# =========================
# Parity plot: P_dew
# =========================
xmin_dew = min(minimum(y_test_dew), minimum(yhat_test_dew))
xmax_dew = max(maximum(y_test_dew), maximum(yhat_test_dew))

p_dew = scatter(
    y_test_dew, yhat_test_dew,
    xlabel = "Actual P_dew",
    ylabel = "Predicted P_dew",
    label = "Test data",
    markersize = 4,
    title = "Parity Plot for P_dew",
    legend = :topleft
)

plot!(
    p_dew,
    [xmin_dew, xmax_dew],
    [xmin_dew, xmax_dew],
    label = "y = x",
    lw = 2
)

savefig(p_dew, "outputs/parity_plot_P_dew.png")
println("Saved: outputs/parity_plot_P_dew.png")